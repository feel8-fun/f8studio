import { studioEvents } from '../api/eventStream';
import { Save } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { editor } from 'monaco-editor';

import { analyzeEditorSession, closeEditorSession, createEditorSession, fetchProject, patchProject, updateEditorSession } from '../api/client';
import type { ProjectRecord, StateSpec } from '../api/contracts';
import { monaco } from './monaco';
import { usePythonLanguageFeatures } from './usePythonLanguageFeatures';

interface Target {
  readonly projectId: string;
  readonly nodeId: string;
  readonly fieldName: string;
}

function readTarget(): Target | null {
  const params = new URLSearchParams(window.location.search);
  const projectId = params.get('project');
  const nodeId = params.get('node');
  const fieldName = params.get('field');
  return projectId && nodeId && fieldName ? { projectId, nodeId, fieldName } : null;
}

function currentField(record: ProjectRecord, target: Target): { field: StateSpec; value: string; nodeName: string; writable: boolean } | null {
  const node = record.document.nodes.find((item) => item.nodeId === target.nodeId);
  const field = node?.spec.stateFields?.find((item) => item.name === target.fieldName);
  if (node === undefined || field === undefined || field.control?.kind !== 'code') return null;
  const raw = node.stateValues[field.name] ?? field.valueSchema.default ?? '';
  if (typeof raw !== 'string') return null;
  const connected = record.document.edges.some((edge) => edge.toNodeId === node.nodeId && edge.kind === 'state' &&
    node.ports.some((port) => port.portId === edge.toPortId && port.kind === 'state' && port.name === field.name));
  return { field, value: raw, nodeName: node.name, writable: field.access !== 'ro' && !connected };
}

export function CodeStateWorkspace() {
  const target = useRef(readTarget()).current;
  const hostRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const sessionRef = useRef<{ id: string; version: number; text: string } | null>(null);
  const syncTailRef = useRef<Promise<void>>(Promise.resolve());
  const analysisTimerRef = useRef<number | null>(null);
  const analysisRunningRef = useRef(false);
  const analysisQueuedRef = useRef(false);
  const analysisGenerationRef = useRef(0);
  const analyzeRef = useRef<() => Promise<void>>(async () => {});
  const saveRef = useRef<() => Promise<void>>(async () => {});
  const baseRef = useRef('');
  const savingRef = useRef(false);
  const [record, setRecord] = useState<ProjectRecord | null>(null);
  const [name, setName] = useState('');
  const [language, setLanguage] = useState('python');
  const [dirty, setDirty] = useState(false);
  const [writable, setWritable] = useState(false);
  const [valid, setValid] = useState(true);
  const [exclusive, setExclusive] = useState(false);
  const [status, setStatus] = useState('Loading code...');

  useEffect(() => {
    if (target === null) { setStatus('Missing code field address'); return; }
    if (navigator.locks === undefined) { setStatus('This browser cannot lock the code editor'); return; }
    let release: (() => void) | null = null;
    let disposed = false;
    const lockName = `f8studio:code:${target.projectId}:${target.nodeId}:${target.fieldName}`;
    void navigator.locks.request(lockName, { ifAvailable: true }, async (lock) => {
      if (disposed) return;
      if (lock === null) { setStatus('This code field is already open in another window'); return; }
      setExclusive(true);
      await new Promise<void>((resolve) => { release = resolve; });
    }).catch((reason: unknown) => setStatus(reason instanceof Error ? reason.message : 'Editor lock failed'));
    return () => { disposed = true; release?.(); };
  }, [target]);

  useEffect(() => {
    if (target === null || !exclusive) return;
    let disposed = false;
    const refresh = async () => {
      try {
        const latest = await fetchProject(target.projectId);
        if (disposed) return;
        const current = currentField(latest, target);
        if (current === null) {
          setValid(false);
          setStatus('The project, node, or code field no longer exists');
          return;
        }
        setRecord(latest);
        setName(`${latest.name} / ${current.nodeName} / ${target.fieldName}`);
        setLanguage(current.field.control?.language ?? 'python');
        setWritable(current.writable);
        setValid(true);
        if (editorRef.current !== null && current.value !== baseRef.current) {
          if (editorRef.current.getValue() === baseRef.current) {
            editorRef.current.setValue(current.value);
            baseRef.current = current.value;
            setDirty(false);
            setStatus('Updated from graph');
          } else setStatus('Code changed elsewhere; review before saving');
        } else if (editorRef.current === null) {
          baseRef.current = current.value;
          setStatus('Ready');
        }
      } catch (reason) {
        if (!disposed) {
          setValid(false);
          setStatus(reason instanceof Error ? reason.message : 'Project unavailable');
        }
      }
    };
    void refresh();
    const unsubscribe = studioEvents.subscribe((event) => {
      if (event.scope === `project:${target.projectId}` && ['graph.committed', 'project.deleted'].includes(event.type)) void refresh();
    }, () => void refresh());
    return () => { disposed = true; unsubscribe(); };
  }, [exclusive, target]);

  const loaded = record !== null;
  useEffect(() => {
    if (!loaded || hostRef.current === null || editorRef.current !== null) return;
    const instance = monaco.editor.create(hostRef.current, {
      value: baseRef.current, language, theme: 'f8studio-dark', automaticLayout: true,
      minimap: { enabled: false }, fontSize: 13, lineHeight: 21, scrollBeyondLastLine: false,
      readOnly: !writable, padding: { top: 10 },
      tabSize: 4,
      quickSuggestions: { other: true, comments: false, strings: false },
      suggestOnTriggerCharacters: true,
    });
    editorRef.current = instance;
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => void saveRef.current());
    const change = instance.onDidChangeModelContent(() => {
      setDirty(instance.getValue() !== baseRef.current);
      analysisGenerationRef.current += 1;
      analysisQueuedRef.current = true;
      const model = instance.getModel();
      if (model !== null) monaco.editor.setModelMarkers(model, 'f8studio', []);
      if (analysisTimerRef.current !== null) window.clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = window.setTimeout(() => {
        analysisTimerRef.current = null;
        void analyzeRef.current();
      }, 700);
    });
    analysisTimerRef.current = window.setTimeout(() => {
      analysisTimerRef.current = null;
      void analyzeRef.current();
    }, 700);
    return () => {
      if (analysisTimerRef.current !== null) window.clearTimeout(analysisTimerRef.current);
      change.dispose(); instance.dispose(); editorRef.current = null;
    };
  }, [loaded]);

  useEffect(() => { editorRef.current?.updateOptions({ readOnly: !writable || !valid }); }, [valid, writable]);

  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', beforeUnload);
    return () => window.removeEventListener('beforeunload', beforeUnload);
  }, [dirty]);

  useEffect(() => () => {
    void syncTailRef.current.then(async () => {
      const session = sessionRef.current;
      if (session !== null) await closeEditorSession(session.id);
    }).catch((error: unknown) => console.error('Failed to close code session', error));
  }, []);

  const syncSession = useCallback(async (): Promise<string> => {
    const operation = syncTailRef.current.then(async () => {
      const text = editorRef.current?.getValue() ?? '';
      const session = sessionRef.current;
      if (session === null) {
        const created = await createEditorSession(
          language === 'json' ? 'json' : 'python', text, `state.${language === 'json' ? 'json' : 'py'}`,
          target === null ? undefined : { projectId: target.projectId, nodeId: target.nodeId, fieldName: target.fieldName },
        );
        sessionRef.current = { id: created.sessionId, version: created.version, text };
        return created.sessionId;
      }
      if (session.text !== text) {
        const updated = await updateEditorSession(session.id, session.version + 1, text);
        sessionRef.current = { id: session.id, version: updated.version, text };
      }
      return session.id;
    });
    syncTailRef.current = operation.then(() => undefined, () => undefined);
    return operation;
  }, [language, target]);
  usePythonLanguageFeatures(loaded && exclusive && valid && language === 'python', syncSession, editorRef);

  const analyze = async () => {
    if (analysisRunningRef.current || editorRef.current === null || !exclusive || !valid) return;
    analysisRunningRef.current = true;
    analysisQueuedRef.current = false;
    const generation = analysisGenerationRef.current;
    try {
      const result = await analyzeEditorSession(await syncSession());
      if (generation !== analysisGenerationRef.current || editorRef.current === null) return;
      const model = editorRef.current?.getModel();
      if (model !== null && model !== undefined) monaco.editor.setModelMarkers(model, 'f8studio', result.diagnostics.map((item) => ({
        severity: item.severity === 'error' ? monaco.MarkerSeverity.Error : item.severity === 'warning' ? monaco.MarkerSeverity.Warning : monaco.MarkerSeverity.Info,
        message: item.message, source: item.source, code: item.rule ?? undefined,
        startLineNumber: item.range.start.line + 1, startColumn: item.range.start.column + 1,
        endLineNumber: item.range.end.line + 1, endColumn: item.range.end.column + 1,
      })));
    } catch (reason) {
      if (generation === analysisGenerationRef.current) setStatus(reason instanceof Error ? `Analysis unavailable: ${reason.message}` : 'Analysis unavailable');
    } finally {
      analysisRunningRef.current = false;
      if (analysisQueuedRef.current && analysisTimerRef.current === null) void analyzeRef.current();
    }
  };
  analyzeRef.current = analyze;

  const save = async () => {
    if (target === null || !exclusive || !valid || !writable || savingRef.current) return;
    const text = editorRef.current?.getValue();
    if (text === undefined) return;
    savingRef.current = true;
    setStatus('Saving...');
    try {
      const latest = await fetchProject(target.projectId);
      const current = currentField(latest, target);
      if (current === null || !current.writable) {
        setValid(false);
        setStatus('The code field was removed or is no longer editable');
        return;
      }
      if (current.value !== baseRef.current) {
        setStatus('Code changed elsewhere; your draft was not overwritten');
        return;
      }
      const result = await patchProject(target.projectId, latest.document, [{ op: 'setNodeState', nodeId: target.nodeId, field: target.fieldName, value: text }]);
      baseRef.current = text;
      setRecord({ ...latest, document: result.document });
      setDirty(false);
      setStatus(result.runtimeErrors.length > 0 ? `Saved; runtime sync failed: ${result.runtimeErrors.join('; ')}` : 'Saved');
    } catch (reason) { setStatus(reason instanceof Error ? reason.message : 'Save failed'); }
    finally { savingRef.current = false; }
  };
  saveRef.current = save;

  return <section className="code-workspace code-state-workspace" aria-label="Code state editor">
    <div className="tool-strip">
      <strong className="code-state-target" title={name}>{name || 'Code state'}</strong>
      <span className="tool-status" role="status">{status}</span>
      <button type="button" className="command-button primary" disabled={!dirty || !exclusive || !valid || !writable} onClick={() => void save()}><Save size={15} />Save</button>
    </div>
    <div className="code-layout">
      <div className="monaco-host" ref={hostRef} />
    </div>
  </section>;
}
