

import { useCallback, useEffect, useRef, useState } from 'react';

import { deployProject, fetchCatalog, fetchDeployJob, fetchLatestDeployment, restartProjectService, stopProject } from '../api/client';
import { isDeployJob } from '../api/contracts';
import { studioEvents } from '../api/eventStream';

import type { CatalogSnapshot, DeployJob, ProjectRecord } from '../api/contracts';

import { errorMessage } from './workspaceUtils';
export function useProjectDeployment({ project, busy, setBusy, setError, setCatalog, setDeployment, refreshingCatalog, reportCommand }: {
  readonly project: ProjectRecord | null;
  readonly busy: boolean;
  readonly setBusy: (value: boolean) => void;
  readonly setError: (value: string | null) => void;
  readonly setCatalog: (value: CatalogSnapshot) => void;
  readonly setDeployment: (value: DeployJob | null) => void;
  readonly refreshingCatalog: boolean;
  readonly reportCommand: (kind: 'success' | 'error', title: string, detail: string) => void;
}) {
  const [stopping, setStopping] = useState(false);
  const stoppingRef = useRef(false);
  const deploymentAbortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    stoppingRef.current = false;
    return () => {
      stoppingRef.current = true;
      deploymentAbortRef.current?.abort();
    };
  }, []);
  const followDeployment = useCallback(async (initialJob: DeployJob): Promise<DeployJob> => {
    if (stoppingRef.current) throw new DOMException("Deployment observer closed", "AbortError");
    deploymentAbortRef.current?.abort();
    const controller = new AbortController();
    deploymentAbortRef.current = controller;
    const job = await new Promise<DeployJob>((resolve, reject) => {
      let finished = false;
      let unsubscribe: () => void = () => {};
      const cancel = () => {
        if (finished) return;
        finished = true;
        unsubscribe();
        reject(new DOMException('Deployment observer closed', 'AbortError'));
      };
      controller.signal.addEventListener('abort', cancel, { once: true });
      const cleanup = () => {
        unsubscribe();
        controller.signal.removeEventListener('abort', cancel);
      };
      const update = (next: DeployJob) => {
        if (finished || next.jobId !== initialJob.jobId) return;
        if (!stoppingRef.current) setDeployment(next);
        if (next.status !== 'queued' && next.status !== 'running') {
          finished = true;
          cleanup();
          resolve(next);
        }
      };
      unsubscribe = studioEvents.subscribe((event) => {
        if (event.type.startsWith('deploy.') && isDeployJob(event.payload)) update(event.payload);
      }, () => {
        void fetchDeployJob(initialJob.jobId).then(update, (reason: unknown) => {
          if (!finished) { finished = true; cleanup(); reject(reason); }
        });
      });
      update(initialJob);
    });
    if (job.status === 'failed' || job.status === 'partially_failed') throw new Error(job.errorMessage || `Deployment ${job.status}`);
    return job;
  }, []);

  const restartService = useCallback(async (serviceId: string) => {
    if (project === null || busy || refreshingCatalog) return;
    setBusy(true);
    setError(null);
    try {
      const initialJob = await restartProjectService(project.projectId, serviceId);
      setCatalog(await fetchCatalog());
      await followDeployment(initialJob);
      reportCommand('success', 'Service restarted', `${serviceId} restarted, node catalog refreshed, and project deployed`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, followDeployment, project, refreshingCatalog]);

  const deploy = useCallback(async () => {
    if (project === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      await followDeployment(await deployProject(project.projectId, project.document.graphRevision));
      if (stoppingRef.current) return;
    } catch (reason) {
      if (!stoppingRef.current) setError(errorMessage(reason));
    } finally {
      if (!stoppingRef.current) setBusy(false);
    }
  }, [busy, followDeployment, project]);

  const stop = useCallback(async () => {
    if (project === null || stopping) return;
    stoppingRef.current = true;
    setStopping(true);
    setBusy(true);
    setError(null);
    try {
      await stopProject(project.projectId);
      setDeployment(await fetchLatestDeployment(project.projectId));
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
      setStopping(false);
      stoppingRef.current = false;
    }
  }, [project, stopping]);

  return { stopping, deploy, stop, restartService };
}
