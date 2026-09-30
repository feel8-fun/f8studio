import { useCallback, useEffect, useRef, useState } from 'react';

import { changeHistory, createProject, deleteProject, exportProjectGraph, fetchCatalog, fetchLatestDeployment, fetchProject, fetchProjects, importProjectGraph, patchProject, refreshCatalog } from '../api/client';
import { isStudioDocument } from '../api/contracts';
import { studioEvents } from '../api/eventStream';

import type { CatalogSnapshot, DeployJob, GraphOperation, ProjectRecord, ProjectSummary } from '../api/contracts';

import { MutationQueue } from './MutationQueue';

import { errorMessage, documentIsNewer } from './workspaceUtils';
const SELECTED_PROJECT_KEY = 'f8studio.selectedProjectId';

export function useGraphProject(resetSelection: () => void, reportCommand: (kind: 'success' | 'error', title: string, detail: string) => void) {
  const [projects, setProjects] = useState<readonly ProjectSummary[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [catalog, setCatalog] = useState<CatalogSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [refreshingCatalog, setRefreshingCatalog] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deployment, setDeployment] = useState<DeployJob | null>(null);
  const [mutations] = useState(() => new MutationQueue());
  const projectRef = useRef<ProjectRecord | null>(null);
  const projectId = project?.projectId ?? null;
  projectRef.current = project;

  const reloadProject = useCallback(async (projectId: string) => {
    const [loaded, latestDeployment] = await Promise.all([
      fetchProject(projectId),
      fetchLatestDeployment(projectId),
    ]);
    setSelectedProjectId(projectId);
    setProject(loaded);
    setDeployment(latestDeployment);
    return loaded;
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      try {
        const [availableProjects, loadedCatalog] = await Promise.all([
          fetchProjects(controller.signal),
          fetchCatalog(controller.signal),
        ]);
        setProjects(availableProjects);
        setCatalog(loadedCatalog);
        const rememberedId = localStorage.getItem(SELECTED_PROJECT_KEY);
        const initial = availableProjects.find((item) => item.projectId === rememberedId) ?? availableProjects.at(-1);
        if (initial !== undefined) {
          setSelectedProjectId(initial.projectId);
          const [record, latestDeployment] = await Promise.all([
            fetchProject(initial.projectId, controller.signal),
            fetchLatestDeployment(initial.projectId, controller.signal),
          ]);
          setProject(record);
          setDeployment(latestDeployment);
        }
      } catch (reason) {
        if (!controller.signal.aborted) setError(errorMessage(reason));
      }
    };
    void load();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (selectedProjectId !== null) localStorage.setItem(SELECTED_PROJECT_KEY, selectedProjectId);
  }, [selectedProjectId]);

  useEffect(() => {
    if (projectId === null) return;
    let disposed = false;
    const refresh = () => {
        void fetchProject(projectId).then((record) => {
          const current = projectRef.current;
          if (!disposed && (current === null || documentIsNewer(record.document, current.document))) {
            setProject(record);
          }
        }, (reason: unknown) => {
          if (!disposed) setError((current) => current ?? errorMessage(reason));
        });
    };
    const unsubscribe = studioEvents.subscribe((envelope) => {
        if (envelope.type === 'project.deleted' && envelope.scope === `project:${projectId}`) {
          setProjects((current) => current.filter((item) => item.projectId !== projectId));
          setProject(null);
          setSelectedProjectId(null);
          setDeployment(null);
          localStorage.removeItem(SELECTED_PROJECT_KEY);
          resetSelection();
          void fetchProjects().then(async (available) => {
            setProjects(available);
            const next = available[0];
            if (next !== undefined) {
              setSelectedProjectId(next.projectId);
              await reloadProject(next.projectId);
            }
          }).catch((reason: unknown) => setError(errorMessage(reason)));
          return;
        }
        if (envelope.type !== 'graph.committed' || envelope.scope !== `project:${projectId}` ||
          typeof envelope.payload !== 'object' || envelope.payload === null) return;
        const payload = envelope.payload as Record<string, unknown>;
        const document = payload.document;
        if (!isStudioDocument(document)) return;
        setProject((current) => {
          if (current === null || current.projectId !== projectId ||
            !documentIsNewer(document, current.document)) return current;
          return { ...current, document };
        });
    }, refresh);
    return () => { disposed = true; unsubscribe(); };
  }, [projectId, reloadProject, resetSelection]);

  const mutateProject = useCallback((targetId: string, operation: (current: ProjectRecord) => Promise<ProjectRecord>): Promise<void> => {
    setSaving(true);
    setError(null);
    const result = mutations.enqueue(async () => {
      const current = projectRef.current;
      if (current === null || current.projectId !== targetId) throw new Error('The selected project changed before this edit could be saved.');
      try {
        const updated = await operation(current);
        const latest = projectRef.current;
        if (latest?.projectId === targetId && !documentIsNewer(latest.document, updated.document)) {
          projectRef.current = updated;
          setProject(updated);
        }
      } catch (reason) {
        // Refresh before queued work is released, without switching the selected project.
        const loaded = await fetchProject(targetId);
        if (projectRef.current?.projectId === targetId) {
          projectRef.current = loaded;
          setProject(loaded);
        }
        throw reason;
      }
    });
    // Attach an error observer for fire-and-forget UI callbacks. Return the original
    // promise so editors that await persistence still receive the failure.
    void result.then(() => {
      setSaving(mutations.pending > 0);
    }, (reason: unknown) => {
      setSaving(mutations.pending > 0);
      setError(errorMessage(reason));
    });
    return result;
  }, [mutations]);

  const commit = useCallback((operations: readonly GraphOperation[]): Promise<void> => {
    if (operations.length === 0) return Promise.resolve();
    return mutateProject(project?.projectId ?? '', async (current) => {
      const result = await patchProject(current.projectId, current.document, operations);
      if (result.runtimeErrors.length > 0) setError(`Saved to project, but runtime sync failed: ${result.runtimeErrors.join('; ')}`);
      return { ...current, document: result.document };
    });
  }, [mutateProject, project?.projectId]);

  const selectProject = useCallback(async (projectId: string) => {
    setBusy(true);
    setError(null);
    setSelectedProjectId(projectId);
    setProject(null);
    setDeployment(null);
    resetSelection();
    try {
      await reloadProject(projectId);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [reloadProject, resetSelection]);

  const addProject = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await createProject(`Untitled ${projects.length + 1}`);
      setProjects(await fetchProjects());
      setSelectedProjectId(created.projectId);
      setProject(created);
      setDeployment(null);
      resetSelection();
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [projects.length, resetSelection]);

  const removeProject = useCallback(async () => {
    const selected = projects.find((item) => item.projectId === selectedProjectId);
    if (selected === undefined || busy || saving || mutations.pending > 0) return;
    if (!window.confirm(`Delete project "${selected.name}" and its saved versions, deployments, and agent sessions? This cannot be undone.`)) return;
    setBusy(true);
    setError(null);
    try {
      await deleteProject(selected.projectId);
      setProjects((current) => current.filter((item) => item.projectId !== selected.projectId));
      setProject(null);
      setSelectedProjectId(null);
      setDeployment(null);
      localStorage.removeItem(SELECTED_PROJECT_KEY);
      const available = await fetchProjects();
      setProjects(available);
      const next = available[0];
      if (next !== undefined) {
        await selectProject(next.projectId);
      }
      resetSelection();
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, projects, saving, selectProject, selectedProjectId, resetSelection, mutations]);

  const history = useCallback(async (action: 'undo' | 'redo') => {
    if (project === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      await mutateProject(project.projectId, async (current) => {
        const result = await changeHistory(current.projectId, action, current.document);
        return { ...current, document: result.document };
      });
    } catch (reason) {
      setError(errorMessage(reason));
      await reloadProject(project.projectId);
    } finally {
      setBusy(false);
    }
  }, [busy, mutateProject, project, reloadProject]);

  const downloadGraph = useCallback(async () => {
    if (project === null || busy) return;
    setError(null);
    try {
      const content = await exportProjectGraph(project.projectId);
      const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `${project.projectId}.f8graph.json`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  }, [busy, project]);

  const uploadGraph = useCallback(async (file: File) => {
    if (project === null || busy) return;
    if (!window.confirm(`Replace the graph in ${project.name} with ${file.name}?`)) return;
    setBusy(true);
    setError(null);
    try {
      const content = await file.text();
      await mutateProject(project.projectId, (current) => importProjectGraph(current.projectId, content, current.document));
      resetSelection();
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, mutateProject, project, resetSelection]);

  const refreshNodeCatalog = useCallback(async () => {
    if (busy || refreshingCatalog) return;
    setRefreshingCatalog(true);
    setError(null);
    try {
      const updated = await refreshCatalog();
      setCatalog(updated);
      reportCommand('success', 'Node catalog refreshed', `${updated.operators.length} operators available`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setRefreshingCatalog(false);
    }
  }, [busy, refreshingCatalog, reportCommand]);

  return { projects, selectedProjectId, project, catalog, busy, setBusy, saving, error, setError,
    deployment, setDeployment, setCatalog, refreshingCatalog, reloadProject, commit,
    selectProject, addProject, removeProject, history, downloadGraph, uploadGraph, refreshNodeCatalog };
}
