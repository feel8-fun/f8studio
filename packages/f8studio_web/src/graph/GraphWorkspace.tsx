import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type FinalConnectionState,
  type Node,
  type OnNodeDrag,
  type ResizeParams,
} from '@xyflow/react';
import { Braces, Check, Copy, Keyboard, Play, Plus, Redo2, RotateCcw, Search, Square, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  ApiError,
  changeHistory,
  createCatalogNode,
  createProject,
  deployProject,
  fetchCatalog,
  fetchDeployJob,
  fetchLatestDeployment,
  fetchHotkeys,
  fetchProject,
  fetchProjects,
  fetchRuntimeMonitors,
  fetchRuntimeNodeState,
  patchProject,
  registerHotkey,
  stopRuntimeService,
  unregisterHotkey,
} from '../api/client';
import { isStudioDocument } from '../api/contracts';
import type {
  CatalogSnapshot,
  DeployJob,
  GraphEdge,
  GraphNode,
  GraphOperation,
  GraphPort,
  HotkeyBinding,
  JsonValue,
  OperatorSpec,
  ProjectRecord,
  ProjectSummary,
  RuntimeMonitor,
  RuntimeStateField,
  ServiceSpec,
  StateSpec,
} from '../api/contracts';
import { connectionError, edgeKindForPort } from './connectionRules';
import {
  absoluteFlowPosition,
  COMPACT_SERVICE_WIDTH,
  compactServiceHeight,
  constrainOperatorPosition,
  duplicateFragment,
  OPERATOR_MIN_HEIGHT,
  OPERATOR_WIDTH,
  operatorHeight,
  projectDocument,
  reconcileProjectedEdges,
  reconcileProjectedNodes,
  SERVICE_MIN_HEIGHT,
  SERVICE_WIDTH,
  serviceChildInsetY,
  type StudioFlowNode,
} from './projection';
import { StateFieldControl } from './StateFieldControl';
import { GraphNodeInteractionContext, StudioNodeView } from './StudioNodeView';

const nodeTypes = { studio: StudioNodeView };
const SELECTED_PROJECT_KEY = 'f8studio.selectedProjectId';
const STUDIO_SERVICE_CLASS = 'f8.pystudio';
const STUDIO_SERVICE_ID = 'studio';

function hotkeyEligible(field: StateSpec): boolean {
  if (field.access !== 'rw') return false;
  const control = (field.uiControl ?? '').split('[', 1)[0]?.trim().toLowerCase() ?? '';
  if (control === 'button') return field.valueSchema.type === 'integer' || field.valueSchema.type === 'number';
  return ['select', 'dropdown', 'dropbox', 'combo', 'combobox'].includes(control) ||
    (field.valueSchema.enum?.length ?? 0) > 0;
}

function HotkeyEditor({ projectId, node, field, disabled }: {
  readonly projectId: string;
  readonly node: GraphNode;
  readonly field: StateSpec;
  readonly disabled: boolean;
}) {
  const [binding, setBinding] = useState<HotkeyBinding | null>(null);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetchHotkeys(projectId).then((bindings) => {
      if (controller.signal.aborted) return;
      const current = bindings.find((item) => item.nodeId === node.nodeId && item.field === field.name) ?? null;
      setBinding(current);
      setDraft(current?.accelerator ?? '');
      setError(null);
    }, (reason: unknown) => {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    });
    return () => controller.abort();
  }, [field.name, node.nodeId, projectId]);

  const save = async () => {
    if (draft.trim() === '') return;
    setBusy(true);
    setError(null);
    try {
      const saved = await registerHotkey({
        accelerator: draft,
        projectId,
        nodeId: node.nodeId,
        field: field.name,
        ...(binding === null ? {} : { bindingId: binding.bindingId }),
      });
      setBinding(saved);
      setDraft(saved.accelerator);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (binding === null) return;
    setBusy(true);
    setError(null);
    try {
      await unregisterHotkey(binding.bindingId);
      setBinding(null);
      setDraft('');
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  return <div className="hotkey-editor">
    <div className="hotkey-heading"><Keyboard size={13} /><span>Global hotkey</span>{binding !== null && <i className={`hotkey-status hotkey-status-${binding.status}`} title={binding.message || binding.status} />}</div>
    <div className="hotkey-input-row">
      <input aria-label={`${field.label ?? field.name} global hotkey`} value={draft} placeholder="Ctrl+Alt+P" disabled={disabled || busy} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void save(); } }} />
      <button className="icon-button bordered" type="button" aria-label={`Save ${field.label ?? field.name} global hotkey`} title="Save global hotkey" disabled={disabled || busy || draft.trim() === ''} onClick={() => void save()}><Check size={14} /></button>
      <button className="icon-button bordered" type="button" aria-label={`Clear ${field.label ?? field.name} global hotkey`} title="Clear global hotkey" disabled={disabled || busy || binding === null} onClick={() => void remove()}><X size={14} /></button>
    </div>
    {error !== null && <small role="alert">{error}</small>}
  </div>;
}

function NodeSchemaEditor({ node, busy, commit }: {
  readonly node: GraphNode;
  readonly busy: boolean;
  readonly commit: (operations: readonly GraphOperation[]) => Promise<void>;
}) {
  const [text, setText] = useState(() => JSON.stringify({ spec: node.spec, ports: node.ports }, null, 2));
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { setText(JSON.stringify({ spec: node.spec, ports: node.ports }, null, 2)); setError(null); }, [node]);
  const apply = async () => {
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed !== 'object' || parsed === null) throw new Error('Schema document must be an object');
      const value = parsed as Record<string, unknown>;
      if (typeof value.spec !== 'object' || value.spec === null || !Array.isArray(value.ports)) {
        throw new Error('Schema document requires spec and ports');
      }
      const ports = value.ports as readonly GraphPort[];
      const replacement: GraphNode = node.kind === 'operator'
        ? { ...node, spec: value.spec as OperatorSpec, ports }
        : { ...node, spec: value.spec as ServiceSpec, ports };
      await commit([{ op: 'replaceNode', node: replacement }]);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : 'Schema update failed');
    }
  };
  return <details className="node-schema-editor">
    <summary><Braces size={14} />Schema</summary>
    <textarea value={text} onChange={(event) => setText(event.target.value)} disabled={busy} spellCheck={false} aria-label="Node schema JSON" />
    {error !== null && <p role="alert">{error}</p>}
    <button className="command-button" type="button" disabled={busy} onClick={() => void apply()}>Apply schema</button>
  </details>;
}

function NodeInspector({
  projectId,
  node,
  services,
  monitor,
  busy,
  commit,
  bindService,
  connectedStateInputs,
}: {
  readonly projectId: string;
  readonly node: GraphNode;
  readonly services: readonly GraphNode[];
  readonly monitor: RuntimeMonitor | null;
  readonly busy: boolean;
  readonly commit: (operations: readonly GraphOperation[]) => Promise<void>;
  readonly bindService: (nodeId: string, serviceId: string) => void;
  readonly connectedStateInputs: ReadonlySet<string>;
}) {
  const fields = node.spec.stateFields ?? [];
  const [runtimeValues, setRuntimeValues] = useState<Readonly<Record<string, RuntimeStateField>>>({});
  const reportedStateError = useRef(false);
  const readonlyFieldNames = useMemo(
    () => fields.filter((field) => field.access === 'ro' && field.name !== 'svcId' && field.name !== 'operatorId')
      .map((field) => field.name),
    [fields],
  );
  const readonlyFieldKey = readonlyFieldNames.join('\u0000');
  useEffect(() => {
    const controller = new AbortController();
    const names = readonlyFieldKey === '' ? [] : readonlyFieldKey.split('\u0000');
    const load = async () => {
      if (names.length === 0) {
        setRuntimeValues({});
        return;
      }
      try {
        const state = await fetchRuntimeNodeState(node.serviceId, node.nodeId, names, controller.signal);
        if (!controller.signal.aborted) {
          setRuntimeValues(Object.fromEntries(state.fields.map((field) => [field.field, field])));
          reportedStateError.current = false;
        }
      } catch (reason: unknown) {
        if (controller.signal.aborted) return;
        setRuntimeValues({});
        if (!reportedStateError.current) {
          reportedStateError.current = true;
          console.error(`Failed to read runtime state for ${node.nodeId}`, reason);
        }
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), 1500);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [node.nodeId, node.serviceId, readonlyFieldKey]);

  const runtimeValue = (field: StateSpec): RuntimeStateField | undefined => {
    if (field.access !== 'ro') return undefined;
    if (field.name === 'svcId') {
      return { field: field.name, found: true, value: node.serviceId, tsMs: null };
    }
    if (field.name === 'operatorId') {
      return { field: field.name, found: true, value: node.nodeId, tsMs: null };
    }
    return runtimeValues[field.name] ?? { field: field.name, found: false, value: null, tsMs: null };
  };
  return <>
    <label className="inspector-field"><span>Name</span><input key={`${node.nodeId}:${node.name}`} disabled={busy} defaultValue={node.name} onBlur={(event) => {
      const name = event.target.value.trim();
      if (name !== '' && name !== node.name) void commit([{ op: 'renameNode', nodeId: node.nodeId, name }]);
    }} /></label>
    <label className="inspector-check"><input type="checkbox" disabled={busy} checked={node.enabled} onChange={(event) => void commit([{ op: 'setNodeEnabled', nodeId: node.nodeId, enabled: event.target.checked }])} /><span>Enabled</span></label>
    <dl>
      <dt>Kind</dt><dd>{node.kind}</dd>
      <dt>Service</dt><dd>{node.serviceClass}</dd>
      <dt>Binding</dt><dd>{node.serviceId}</dd>
    </dl>
    {node.kind === 'operator' && <label className="inspector-field"><span>Service binding</span><select disabled={busy} value={node.serviceId} onChange={(event) => bindService(node.nodeId, event.target.value)}>{services.filter((service) => service.kind === 'service' && service.serviceClass === node.serviceClass).map((service) => <option key={service.serviceId} value={service.serviceId}>{service.name}</option>)}</select></label>}
    <h2>Runtime</h2>
    {monitor === null ? <p className="monitor-empty">No monitor sample</p> : <dl className="monitor-values">
      <dt>Status</dt><dd>{monitor.alive ? (monitor.ready ? 'Ready' : 'Starting') : 'Offline'}</dd>
      <dt>CPU</dt><dd>{(monitor.cpu?.processPercent ?? 0).toFixed(1)}%</dd>
      <dt>Memory</dt><dd>{((monitor.memory?.rssBytes ?? 0) / 1048576).toFixed(1)} MiB</dd>
      <dt>Queue</dt><dd>{monitor.queue?.depth ?? 0}</dd>
      <dt>Latency p95</dt><dd>{(monitor.timing?.latencyMsP95 ?? 0).toFixed(1)} ms</dd>
    </dl>}
    {fields.length > 0 && <h2>State</h2>}
    <div className="inspector-fields">{fields.map((field) => {
      const connected = connectedStateInputs.has(`${node.nodeId}:${field.name}`);
      return <div className="inspector-state-field" key={field.name}>
        <StateFieldControl
          node={node}
          field={field}
          disabled={busy}
          connected={connected}
          runtimeValue={runtimeValue(field)}
          onCommit={(value) => void commit([{ op: 'setNodeState', nodeId: node.nodeId, field: field.name, value }])}
        />
        {hotkeyEligible(field) && <HotkeyEditor projectId={projectId} node={node} field={field} disabled={busy || connected} />}
      </div>;
    })}</div>
    <NodeSchemaEditor node={node} busy={busy} commit={commit} />
    <button className="danger-command" type="button" disabled={busy} onClick={() => void commit([{ op: 'deleteNode', nodeId: node.nodeId }])}><Trash2 size={15} /> {node.kind === 'service' ? 'Delete service and operators' : 'Delete node'}</button>
  </>;
}

function EdgeInspector({
  edge,
  nodes,
  busy,
  replace,
  remove,
}: {
  readonly edge: GraphEdge;
  readonly nodes: readonly GraphNode[];
  readonly busy: boolean;
  readonly replace: (edge: GraphEdge) => void;
  readonly remove: (edgeId: string) => void;
}) {
  const source = nodes.find((node) => node.nodeId === edge.fromNodeId);
  const target = nodes.find((node) => node.nodeId === edge.toNodeId);
  const sourcePort = source?.ports.find((port) => port.portId === edge.fromPortId);
  const targetPort = target?.ports.find((port) => port.portId === edge.toPortId);
  return <>
    <dl>
      <dt>Kind</dt><dd><span className={`edge-kind edge-kind-${edge.kind}`}>{edge.kind}</span></dd>
      <dt>From</dt><dd>{source?.name ?? edge.fromNodeId}.{sourcePort?.name ?? edge.fromPortId}</dd>
      <dt>To</dt><dd>{target?.name ?? edge.toNodeId}.{targetPort?.name ?? edge.toPortId}</dd>
    </dl>
    {edge.kind === 'data' ? <>
      <label className="inspector-field"><span>Delivery</span><select disabled={busy} value={edge.strategy} onChange={(event) => replace({ ...edge, strategy: event.target.value === 'queue' ? 'queue' : 'latest' })}>
        <option value="latest">Latest value</option>
        <option value="queue">Bounded queue</option>
      </select></label>
      <label className="inspector-field"><span>Queue size</span><input key={`${edge.edgeId}:${edge.queueSize}`} type="number" min={1} step={1} disabled={busy || edge.strategy !== 'queue'} defaultValue={edge.queueSize} onBlur={(event) => {
        const queueSize = Number(event.target.value);
        if (Number.isInteger(queueSize) && queueSize >= 1 && queueSize !== edge.queueSize) replace({ ...edge, queueSize });
      }} /></label>
      <label className="inspector-field"><span>Stale timeout (ms)</span><input key={`${edge.edgeId}:${edge.timeoutMs ?? 'disabled'}`} type="number" min={0} step={1} disabled={busy} defaultValue={edge.timeoutMs ?? ''} placeholder="Disabled" onBlur={(event) => {
        const timeoutMs = event.target.value.trim() === '' ? null : Number(event.target.value);
        if ((timeoutMs === null || (Number.isInteger(timeoutMs) && timeoutMs >= 0)) && timeoutMs !== edge.timeoutMs) {
          replace({ ...edge, timeoutMs });
        }
      }} /></label>
    </> : <p className="edge-policy-note">{edge.kind === 'exec' ? 'Exec order is local to one service.' : 'State propagation is latest-value and cycle checked.'}</p>}
    <button className="danger-command" type="button" disabled={busy} onClick={() => remove(edge.edgeId)}><Trash2 size={15} /> Delete connection</button>
  </>;
}

function newId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replaceAll('-', '')}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown graph operation error';
}

function documentIsNewer(next: ProjectRecord['document'], current: ProjectRecord['document']): boolean {
  return next.graphRevision > current.graphRevision ||
    (next.graphRevision === current.graphRevision && next.layoutRevision > current.layoutRevision);
}

function GraphWorkspaceInner() {
  const [projects, setProjects] = useState<readonly ProjectSummary[]>([]);
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [catalog, setCatalog] = useState<CatalogSnapshot | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [search, setSearch] = useState('');
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [deployment, setDeployment] = useState<DeployJob | null>(null);
  const [monitors, setMonitors] = useState<readonly RuntimeMonitor[]>([]);
  const [projectedProjectId, setProjectedProjectId] = useState<string | null>(null);
  const projectedProjectIdRef = useRef<string | null>(null);
  const fittedProjectId = useRef<string | null>(null);
  const mutationInFlight = useRef(false);
  const connectedStateInputsCache = useRef<ReadonlySet<string>>(new Set());
  const projectRef = useRef<ProjectRecord | null>(null);
  const nodesInitialized = useNodesInitialized();
  const { fitView, screenToFlowPosition } = useReactFlow<StudioFlowNode, Edge>();
  const projectId = project?.projectId ?? null;
  projectRef.current = project;

  const reloadProject = useCallback(async (projectId: string) => {
    const [loaded, latestDeployment] = await Promise.all([
      fetchProject(projectId),
      fetchLatestDeployment(projectId),
    ]);
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
    if (project !== null) localStorage.setItem(SELECTED_PROJECT_KEY, project.projectId);
  }, [project]);

  useEffect(() => {
    if (projectId === null) return;
    let disposed = false;
    let socket: WebSocket | null = null;
    let retry = 0;
    let retryTimer: number | null = null;
    const connect = () => {
      if (disposed) return;
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/events`);
      socket.onopen = () => {
        retry = 0;
        void fetchProject(projectId).then((record) => {
          const current = projectRef.current;
          if (!disposed && (current === null || documentIsNewer(record.document, current.document))) {
            setProject(record);
          }
        }, (reason: unknown) => {
          if (!disposed) setError((current) => current ?? errorMessage(reason));
        });
      };
      socket.onmessage = (event) => {
        let decoded: unknown;
        try { decoded = JSON.parse(String(event.data)); }
        catch (reason) { console.error('Invalid graph event JSON', reason); return; }
        if (typeof decoded !== 'object' || decoded === null) return;
        const envelope = decoded as Record<string, unknown>;
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
      };
      socket.onclose = () => {
        if (!disposed) retryTimer = window.setTimeout(connect, Math.min(5000, 300 * 2 ** retry++));
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => {
      disposed = true;
      socket?.close();
      if (retryTimer !== null) window.clearTimeout(retryTimer);
    };
  }, [projectId]);

  useEffect(() => {
    const controller = new AbortController();
    let timeoutId: number | null = null;
    const refresh = async () => {
      try {
        setMonitors(await fetchRuntimeMonitors(controller.signal));
      } catch (reason) {
        if (!controller.signal.aborted) setError((current) => current ?? errorMessage(reason));
      }
      if (!controller.signal.aborted) timeoutId = window.setTimeout(() => void refresh(), 2000);
    };
    void refresh();
    return () => {
      controller.abort();
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
    fittedProjectId.current = null;
  }, [projectId]);

  useEffect(() => {
    if (project === null) {
      setNodes([]);
      setEdges([]);
      setProjectedProjectId(null);
      projectedProjectIdRef.current = null;
      return;
    }
    const projected = projectDocument(project.document);
    const canReuseProjection = projectedProjectIdRef.current === project.projectId;
    setNodes((current) => canReuseProjection
      ? reconcileProjectedNodes(current, projected.nodes)
      : projected.nodes);
    setEdges((current) => canReuseProjection
      ? reconcileProjectedEdges(current, projected.edges)
      : projected.edges);
    setProjectedProjectId(project.projectId);
    projectedProjectIdRef.current = project.projectId;
  }, [project, setEdges, setNodes]);

  useEffect(() => {
    if (
      projectId === null ||
      projectedProjectId !== projectId ||
      !nodesInitialized ||
      nodes.length === 0 ||
      fittedProjectId.current === projectId
    ) return;
    fittedProjectId.current = projectId;
    void fitView({ padding: 0.2, maxZoom: 1, duration: 0 });
  }, [fitView, nodes.length, nodesInitialized, projectId, projectedProjectId]);

  const commit = useCallback(async (operations: readonly GraphOperation[]) => {
    if (project === null || busy || mutationInFlight.current || operations.length === 0) return;
    mutationInFlight.current = true;
    setSaving(true);
    setError(null);
    try {
      const result = await patchProject(project.projectId, project.document, operations);
      setProject((current) => {
        const base = current?.projectId === project.projectId ? current : project;
        if (documentIsNewer(base.document, result.document)) return base;
        if (base.document.graphRevision === result.document.graphRevision &&
          base.document.layoutRevision === result.document.layoutRevision) return base;
        return { ...base, document: result.document };
      });
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 409) {
        await reloadProject(project.projectId);
        setError('The graph changed elsewhere. Reloaded the latest revision.');
      } else {
        setError(errorMessage(reason));
        await reloadProject(project.projectId);
      }
    } finally {
      mutationInFlight.current = false;
      setSaving(false);
    }
  }, [busy, project, reloadProject]);

  const restoreProjection = useCallback(() => {
    if (project === null) return;
    const projected = projectDocument(project.document);
    setNodes((current) => reconcileProjectedNodes(current, projected.nodes));
    setEdges((current) => reconcileProjectedEdges(current, projected.edges));
  }, [project, setEdges, setNodes]);

  const bindOperatorService = useCallback((nodeId: string, serviceId: string) => {
    if (project === null || busy) return;
    const projected = projectDocument(project.document);
    const operator = projected.nodes.find((node) => node.id === nodeId);
    const service = projected.nodes.find((node) => node.id === serviceId && node.data.graphNode.kind === 'service');
    if (operator === undefined || service === undefined) {
      setError('The selected operator or service is no longer available.');
      return;
    }
    const width = typeof service.style?.width === 'number' ? service.style.width : SERVICE_WIDTH;
    const height = typeof service.style?.height === 'number' ? service.style.height : SERVICE_MIN_HEIGHT;
    const relative = constrainOperatorPosition(
      operator.position,
      width,
      height,
      operatorHeight(operator.data.graphNode),
      serviceChildInsetY(service.data.graphNode),
    );
    const currentLayout = project.document.layout.find((layout) => layout.nodeId === nodeId);
    void commit([
      { op: 'bindOperatorService', nodeId, serviceId },
      {
        op: 'setNodeLayout',
        layout: {
          nodeId,
          x: service.position.x + relative.x,
          y: service.position.y + relative.y,
          width: currentLayout?.width,
          height: currentLayout?.height,
          collapsed: currentLayout?.collapsed ?? false,
        },
      },
    ]);
  }, [busy, commit, project]);

  const selectProject = useCallback(async (projectId: string) => {
    setBusy(true);
    setError(null);
    try {
      await reloadProject(projectId);
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [reloadProject]);

  const addProject = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await createProject(`Untitled ${projects.length + 1}`);
      setProjects(await fetchProjects());
      setProject(created);
      setDeployment(null);
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [projects.length]);

  const addSpec = useCallback(async (spec: ServiceSpec | OperatorSpec) => {
    if (project === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      const nodeId = newId(spec.specKind === 'service' ? 'service' : 'operator');
      const nodesToCreate: GraphNode[] = [];
      let selectedNode: GraphNode;
      if (spec.specKind === 'service') {
        selectedNode = await createCatalogNode({ kind: 'service', nodeId, serviceClass: spec.serviceClass });
        nodesToCreate.push(selectedNode);
      } else {
        let service = project.document.nodes.find(
          (node) => node.kind === 'service' && node.serviceClass === spec.serviceClass,
        );
        if (service === undefined) {
          if (spec.serviceClass !== STUDIO_SERVICE_CLASS) {
            throw new Error(`Add a ${spec.serviceClass} service before this operator`);
          }
          if (project.document.nodes.some((node) => node.nodeId === STUDIO_SERVICE_ID)) {
            throw new Error(`Node id ${STUDIO_SERVICE_ID} is reserved for the built-in Studio service`);
          }
          service = await createCatalogNode({
            kind: 'service',
            nodeId: STUDIO_SERVICE_ID,
            serviceClass: STUDIO_SERVICE_CLASS,
          });
          nodesToCreate.push(service);
        }
        selectedNode = await createCatalogNode({
          kind: 'operator',
          nodeId,
          serviceId: service.serviceId,
          serviceClass: spec.serviceClass,
          operatorClass: spec.operatorClass,
        });
        nodesToCreate.push(selectedNode);
      }
      let nextServiceIndex = project.document.nodes.filter((node) => node.kind === 'service').length;
      const operations = nodesToCreate.map((node): GraphOperation => {
        if (node.kind !== 'service') return { op: 'createNode', node };
        const serviceIndex = nextServiceIndex;
        nextServiceIndex += 1;
        return {
          op: 'createNode',
          node,
          layout: {
            nodeId: node.nodeId,
            x: 80 + (serviceIndex % 2) * (SERVICE_WIDTH + 80),
            y: 80 + Math.floor(serviceIndex / 2) * (SERVICE_MIN_HEIGHT + 80),
            width: COMPACT_SERVICE_WIDTH,
            height: compactServiceHeight(node),
            collapsed: false,
          },
        };
      });
      const result = await patchProject(project.projectId, project.document, operations);
      setProject({ ...project, document: result.document });
      setSelectedNodeId(selectedNode.nodeId);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 409) await reloadProject(project.projectId);
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, project, reloadProject]);

  const connect = useCallback((connection: Connection) => {
    if (project === null || connection.sourceHandle === null || connection.targetHandle === null) return;
    const invalid = connectionError(project.document, connection);
    if (invalid !== null) {
      setError(invalid);
      restoreProjection();
      return;
    }
    const sourceNode = project.document.nodes.find((node) => node.nodeId === connection.source);
    const sourcePort = sourceNode?.ports.find((port) => port.portId === connection.sourceHandle);
    if (sourcePort === undefined) return;
    const edge: GraphEdge = {
      edgeId: newId('edge'),
      fromNodeId: connection.source,
      fromPortId: connection.sourceHandle,
      toNodeId: connection.target,
      toPortId: connection.targetHandle,
      kind: edgeKindForPort(sourcePort),
      strategy: 'latest',
      queueSize: 16,
      timeoutMs: null,
    };
    void commit([{ op: 'connectEdge', edge }]);
  }, [commit, project, restoreProjection]);

  const isValidConnection = useCallback((connection: Connection | Edge) => {
    if (project === null) return false;
    return connectionError(project.document, {
      source: connection.source,
      sourceHandle: connection.sourceHandle ?? null,
      target: connection.target,
      targetHandle: connection.targetHandle ?? null,
    }) === null;
  }, [project]);

  const connectionEnded = useCallback((_event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
    if (project === null || state.isValid !== false || state.fromHandle === null || state.toHandle === null) return;
    const fromSource = state.fromHandle.type === 'source';
    const invalid = connectionError(project.document, {
      source: fromSource ? state.fromHandle.nodeId : state.toHandle.nodeId,
      sourceHandle: (fromSource ? state.fromHandle.id : state.toHandle.id) ?? null,
      target: fromSource ? state.toHandle.nodeId : state.fromHandle.nodeId,
      targetHandle: (fromSource ? state.toHandle.id : state.fromHandle.id) ?? null,
    });
    if (invalid !== null) setError(invalid);
  }, [project]);

  const deleteNodes = useCallback((deleted: Node[]) => {
    if (project === null) return;
    const deletedIds = new Set(deleted.map((node) => node.id));
    const deletedServiceIds = new Set(project.document.nodes.flatMap((node) =>
      node.kind === 'service' && deletedIds.has(node.nodeId) ? [node.serviceId] : [],
    ));
    const operations = project.document.nodes.flatMap((node): GraphOperation[] => {
      if (!deletedIds.has(node.nodeId)) return [];
      if (node.kind === 'operator' && deletedServiceIds.has(node.serviceId)) return [];
      return [{ op: 'deleteNode', nodeId: node.nodeId }];
    });
    void commit(operations);
  }, [commit, project]);

  const deleteEdges = useCallback((deleted: Edge[]) => {
    if (selectedEdgeId !== null && deleted.some((edge) => edge.id === selectedEdgeId)) setSelectedEdgeId(null);
    void commit(deleted.map((edge): GraphOperation => ({ op: 'disconnectEdge', edgeId: edge.id })));
  }, [commit, selectedEdgeId]);

  const moveNode: OnNodeDrag<StudioFlowNode> = useCallback((event, node) => {
    if (project === null || busy) return;
    const projected = projectDocument(project.document);
    const graphNode = project.document.nodes.find((candidate) => candidate.nodeId === node.id);
    if (graphNode === undefined) {
      restoreProjection();
      setError(`Node ${node.id} is no longer available.`);
      return;
    }
    if (graphNode.kind === 'service') {
      const previous = projected.nodes.find((candidate) => candidate.id === node.id);
      if (previous === undefined) return;
      const delta = { x: node.position.x - previous.position.x, y: node.position.y - previous.position.y };
      const movedNodeIds = new Set(project.document.nodes.flatMap((candidate) =>
        candidate.nodeId === graphNode.nodeId || (candidate.kind === 'operator' && candidate.serviceId === graphNode.serviceId)
          ? [candidate.nodeId]
          : [],
      ));
      const operations = projected.nodes.flatMap((candidate): GraphOperation[] => {
        if (!movedNodeIds.has(candidate.id)) return [];
        const currentLayout = project.document.layout.find((layout) => layout.nodeId === candidate.id);
        const absolute = candidate.id === node.id
          ? node.position
          : currentLayout === undefined
            ? absoluteFlowPosition(candidate, projected.nodes)
            : { x: currentLayout.x, y: currentLayout.y };
        return [{
          op: 'setNodeLayout',
          layout: {
            nodeId: candidate.id,
            x: absolute.x + (candidate.id === node.id ? 0 : delta.x),
            y: absolute.y + (candidate.id === node.id ? 0 : delta.y),
            width: candidate.id === node.id && typeof candidate.style?.width === 'number'
              ? candidate.style.width
              : currentLayout?.width,
            height: candidate.id === node.id && typeof candidate.style?.height === 'number'
              ? candidate.style.height
              : currentLayout?.height,
            collapsed: currentLayout?.collapsed ?? false,
          },
        }];
      });
      void commit(operations);
      return;
    }

    const draggedNodes = nodes.map((candidate) => candidate.id === node.id ? node : candidate);
    const absolute = absoluteFlowPosition(node, draggedNodes);
    const center = {
      x: absolute.x + (node.measured?.width ?? OPERATOR_WIDTH) / 2,
      y: absolute.y + (node.measured?.height ?? OPERATOR_MIN_HEIGHT) / 2,
    };
    const changedTouch = 'changedTouches' in event ? event.changedTouches.item(0) : null;
    const dropPosition = 'clientX' in event
      ? screenToFlowPosition({ x: event.clientX, y: event.clientY })
      : changedTouch === null
        ? center
        : screenToFlowPosition({ x: changedTouch.clientX, y: changedTouch.clientY });
    const target = projected.nodes.find((candidate) => {
      if (candidate.data.graphNode.kind !== 'service') return false;
      const width = typeof candidate.style?.width === 'number' ? candidate.style.width : SERVICE_WIDTH;
      const height = typeof candidate.style?.height === 'number' ? candidate.style.height : SERVICE_MIN_HEIGHT;
      return dropPosition.x >= candidate.position.x && dropPosition.x <= candidate.position.x + width &&
        dropPosition.y >= candidate.position.y && dropPosition.y <= candidate.position.y + height;
    });
    if (target === undefined) {
      restoreProjection();
      setError('Operators must remain inside a compatible service container.');
      return;
    }
    if (target.data.graphNode.serviceClass !== graphNode.serviceClass) {
      restoreProjection();
      setError(`${graphNode.name} requires ${graphNode.serviceClass}.`);
      return;
    }
    const width = typeof target.style?.width === 'number' ? target.style.width : SERVICE_WIDTH;
    const height = typeof target.style?.height === 'number' ? target.style.height : SERVICE_MIN_HEIGHT;
    const relative = constrainOperatorPosition(
      { x: absolute.x - target.position.x, y: absolute.y - target.position.y },
      width,
      height,
      node.measured?.height ?? OPERATOR_MIN_HEIGHT,
      serviceChildInsetY(target.data.graphNode),
    );
    const currentLayout = project.document.layout.find((layout) => layout.nodeId === node.id);
    const operations: GraphOperation[] = [];
    if (graphNode.serviceId !== target.data.graphNode.serviceId) {
      operations.push({ op: 'bindOperatorService', nodeId: node.id, serviceId: target.data.graphNode.serviceId });
    }
    operations.push({
      op: 'setNodeLayout',
      layout: {
        nodeId: node.id,
        x: target.position.x + relative.x,
        y: target.position.y + relative.y,
        width: currentLayout?.width,
        height: currentLayout?.height,
        collapsed: currentLayout?.collapsed ?? false,
      },
    });
    void commit(operations);
  }, [busy, commit, nodes, project, restoreProjection, screenToFlowPosition]);

  const history = useCallback(async (action: 'undo' | 'redo') => {
    if (project === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await changeHistory(project.projectId, action, project.document);
      setProject({ ...project, document: result.document });
    } catch (reason) {
      setError(errorMessage(reason));
      await reloadProject(project.projectId);
    } finally {
      setBusy(false);
    }
  }, [busy, project, reloadProject]);

  const deploy = useCallback(async () => {
    if (project === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      let job = await deployProject(project.projectId, project.document.graphRevision);
      setDeployment(job);
      for (let attempt = 0; attempt < 100 && (job.status === 'queued' || job.status === 'running'); attempt += 1) {
        await new Promise<void>((resolve) => setTimeout(resolve, 200));
        job = await fetchDeployJob(job.jobId);
        setDeployment(job);
      }
      if (job.status === 'queued' || job.status === 'running') throw new Error('Deployment did not finish within 20 seconds');
      if (job.status !== 'succeeded') setError(job.errorMessage || `Deployment ${job.status}`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, project]);

  const stop = useCallback(async () => {
    if (project === null || busy) return;
    const serviceIds = [...new Set(project.document.nodes.filter((node) => node.kind === 'service').map((node) => node.serviceId))];
    setBusy(true);
    setError(null);
    try {
      await Promise.all(serviceIds.map(stopRuntimeService));
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, project]);

  const duplicateSelection = useCallback(() => {
    if (project === null || busy) return;
    const selectedIds = new Set(nodes.filter((node) => node.selected).map((node) => node.id));
    if (selectedIds.size === 0 && selectedNodeId !== null) selectedIds.add(selectedNodeId);
    const operation = duplicateFragment(project.document, selectedIds, newId);
    if (operation !== null) void commit([operation]);
  }, [busy, commit, nodes, project, selectedNodeId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'd') {
        event.preventDefault();
        duplicateSelection();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [duplicateSelection]);

  const query = search.trim().toLowerCase();
  const services = useMemo(() => (catalog?.services ?? []).filter((spec) => !spec.hiddenInPalette &&
    `${spec.label} ${spec.serviceClass} ${(spec.tags ?? []).join(' ')}`.toLowerCase().includes(query)), [catalog, query]);
  const operators = useMemo(() => (catalog?.operators ?? []).filter((spec) => !spec.hiddenInPalette &&
    `${spec.label} ${spec.operatorClass} ${(spec.tags ?? []).join(' ')}`.toLowerCase().includes(query)), [catalog, query]);
  const selectedNode = project?.document.nodes.find((node) => node.nodeId === selectedNodeId) ?? null;
  const selectedEdge = project?.document.edges.find((edge) => edge.edgeId === selectedEdgeId) ?? null;
  const connectedStateInputs = useMemo(() => {
    if (project === null) {
      const empty = new Set<string>();
      connectedStateInputsCache.current = empty;
      return empty;
    }
    const graphNodes = new Map(project.document.nodes.map((node) => [node.nodeId, node]));
    const next = new Set(project.document.edges.flatMap((edge) => {
      if (edge.kind !== 'state') return [];
      const target = graphNodes.get(edge.toNodeId);
      const port = target?.ports.find((candidate) => candidate.portId === edge.toPortId);
      return port === undefined ? [] : [`${edge.toNodeId}:${port.runtimeName}`];
    }));
    const current = connectedStateInputsCache.current;
    if (current.size === next.size && [...next].every((key) => current.has(key))) return current;
    connectedStateInputsCache.current = next;
    return next;
  }, [project]);
  const replaceEdge = useCallback((edge: GraphEdge) => {
    void commit([
      { op: 'disconnectEdge', edgeId: edge.edgeId },
      { op: 'connectEdge', edge },
    ]);
  }, [commit]);
  const removeEdge = useCallback((edgeId: string) => {
    setSelectedEdgeId(null);
    void commit([{ op: 'disconnectEdge', edgeId }]);
  }, [commit]);
  const commitRef = useRef(commit);
  commitRef.current = commit;
  const resizeServiceRef = useRef<(nodeId: string, bounds: ResizeParams) => void>(() => undefined);
  resizeServiceRef.current = (nodeId, bounds) => {
    if (project === null || busy) return;
    const service = project.document.nodes.find((candidate) => candidate.nodeId === nodeId && candidate.kind === 'service');
    if (service === undefined) {
      restoreProjection();
      setError(`Service ${nodeId} is no longer available.`);
      return;
    }
    const projected = projectDocument(project.document);
    const serviceNode = projected.nodes.find((candidate) => candidate.id === nodeId);
    if (serviceNode === undefined) {
      restoreProjection();
      setError(`Service ${nodeId} has no projected layout.`);
      return;
    }
    const serviceLayout = project.document.layout.find((layout) => layout.nodeId === nodeId);
    const operations: GraphOperation[] = [{
      op: 'setNodeLayout',
      layout: {
        nodeId,
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
        collapsed: serviceLayout?.collapsed ?? false,
      },
    }];
    for (const child of project.document.nodes) {
      if (child.kind !== 'operator' || child.serviceId !== service.serviceId) continue;
      const childNode = projected.nodes.find((candidate) => candidate.id === child.nodeId);
      if (childNode === undefined) continue;
      const relative = constrainOperatorPosition(
        childNode.position,
        bounds.width,
        bounds.height,
        operatorHeight(child),
        serviceChildInsetY(service),
      );
      const childLayout = project.document.layout.find((layout) => layout.nodeId === child.nodeId);
      operations.push({
        op: 'setNodeLayout',
        layout: {
          nodeId: child.nodeId,
          x: bounds.x + relative.x,
          y: bounds.y + relative.y,
          width: childLayout?.width,
          height: childLayout?.height,
          collapsed: childLayout?.collapsed ?? false,
        },
      });
    }
    void commitRef.current(operations);
  };
  const resizeService = useCallback((nodeId: string, bounds: ResizeParams) => {
    resizeServiceRef.current(nodeId, bounds);
  }, []);
  const setNodeState = useCallback((nodeId: string, field: string, value: JsonValue) => {
    void commitRef.current([{ op: 'setNodeState', nodeId, field, value }]);
  }, []);
  const nodeInteraction = useMemo(() => ({
    busy,
    connectedStateInputs,
    resizeService,
    setState: setNodeState,
  }), [busy, connectedStateInputs, resizeService, setNodeState]);
  const selectedMonitor = selectedNode === null ? null : monitors.find((monitor) => monitor.nodeId === selectedNode.nodeId) ??
    (selectedNode.kind === 'service' ? monitors.find((monitor) => monitor.serviceId === selectedNode.serviceId) ?? null : null);
  const locked = busy || saving;

  return (
    <div className="graph-workspace">
      <aside className="graph-palette" aria-label="Node catalog">
        <div className="project-control">
          <label htmlFor="project-select">Project</label>
          <div>
            <select id="project-select" value={project?.projectId ?? ''} disabled={busy} onChange={(event) => void selectProject(event.target.value)}>
              <option value="" disabled>Select project</option>
              {projects.map((item) => <option key={item.projectId} value={item.projectId}>{item.name}</option>)}
            </select>
            <button type="button" className="small-icon-button" title="New project" aria-label="New project" disabled={busy} onClick={() => void addProject()}><Plus size={16} /></button>
          </div>
        </div>
        <label className="catalog-search">
          <Search size={15} />
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search nodes" aria-label="Search nodes" />
        </label>
        <div className="catalog-list">
          <h2>Services</h2>
          {services.map((spec) => <button key={spec.serviceClass} type="button" disabled={busy || project === null} onClick={() => void addSpec(spec)}>
            <strong>{spec.label}</strong><span>{spec.serviceClass}</span>
          </button>)}
          <h2>Operators</h2>
          {operators.map((spec) => {
            const bound = project?.document.nodes.some((node) => node.kind === 'service' && node.serviceClass === spec.serviceClass) ?? false;
            const createsBuiltInService = spec.serviceClass === STUDIO_SERVICE_CLASS;
            return <button key={`${spec.serviceClass}:${spec.operatorClass}`} type="button" disabled={busy || project === null || (!bound && !createsBuiltInService)} title={bound || createsBuiltInService ? spec.description : `Requires ${spec.serviceClass}`} onClick={() => void addSpec(spec)}>
              <strong>{spec.label}</strong><span>{spec.operatorClass}</span>
            </button>;
          })}
        </div>
      </aside>

      <section className="graph-canvas" aria-label="Graph canvas">
        <div className="graph-toolbar">
          <button type="button" title="Undo" aria-label="Undo" disabled={busy || project === null} onClick={() => void history('undo')}><RotateCcw size={16} /></button>
          <button type="button" title="Redo" aria-label="Redo" disabled={busy || project === null} onClick={() => void history('redo')}><Redo2 size={16} /></button>
          <button type="button" title="Duplicate selection" aria-label="Duplicate selection" disabled={busy || project === null || (selectedNodeId === null && !nodes.some((node) => node.selected))} onClick={duplicateSelection}><Copy size={15} /></button>
          <button type="button" title="Deploy" aria-label="Deploy" disabled={busy || project === null} onClick={() => void deploy()}><Play size={16} /></button>
          <button type="button" title="Stop services" aria-label="Stop services" disabled={busy || project === null} onClick={() => void stop()}><Square size={14} /></button>
          <span>{project === null ? 'No project selected' : `Draft r${project.document.graphRevision} · Layout r${project.document.layoutRevision}`}</span>
          <span className={`deploy-state deploy-${deployment?.status ?? 'none'}`}>{deployment === null ? 'Not deployed' : `${deployment.status} r${deployment.sourceGraphRevision}`}</span>
          <span className="save-state">{saving ? 'Saving...' : busy ? 'Working...' : 'Saved'}</span>
        </div>
        {project === null ? <div className="empty-state"><p>Create a project to start building a graph.</p><button className="command-button primary" type="button" onClick={() => void addProject()}>New project</button></div> :
          <GraphNodeInteractionContext.Provider value={nodeInteraction}><ReactFlow<StudioFlowNode, Edge>
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={connect}
            onConnectEnd={connectionEnded}
            isValidConnection={isValidConnection}
            onNodesDelete={deleteNodes}
            onEdgesDelete={deleteEdges}
            onNodeDragStop={moveNode}
            onNodeClick={(_event, node) => {
              setSelectedNodeId(node.id);
              setSelectedEdgeId(null);
            }}
            onEdgeClick={(_event, edge) => {
              setSelectedEdgeId(edge.id);
              setSelectedNodeId(null);
            }}
            onPaneClick={() => {
              setSelectedNodeId(null);
              setSelectedEdgeId(null);
            }}
            nodesDraggable={!busy}
            nodesConnectable={!busy}
            edgesReconnectable={!busy}
            onlyRenderVisibleElements
            fitView
            fitViewOptions={{ maxZoom: 1 }}
            minZoom={0.15}
            maxZoom={2}
            deleteKeyCode={locked ? null : ['Backspace', 'Delete']}
            multiSelectionKeyCode={['Control', 'Meta']}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap pannable zoomable nodeColor={(node) => node.className === 'flow-node-service' ? '#469b79' : '#5d7896'} />
          </ReactFlow></GraphNodeInteractionContext.Provider>}
        {error !== null && <div className="graph-error" role="alert">{error}</div>}
      </section>

      <aside className="graph-inspector" aria-label="Inspector">
        <h2>Inspector</h2>
        {selectedNode !== null && project !== null ? <NodeInspector projectId={project.projectId} node={selectedNode} services={project.document.nodes} monitor={selectedMonitor} busy={busy} commit={commit} bindService={bindOperatorService} connectedStateInputs={connectedStateInputs} /> :
          selectedEdge !== null ? <EdgeInspector edge={selectedEdge} nodes={project?.document.nodes ?? []} busy={busy} replace={replaceEdge} remove={removeEdge} /> :
            <p>Select a node or connection to inspect it.</p>}
        {deployment !== null && deployment.serviceResults.some((result) => !result.success) && <div className="deploy-errors">{deployment.serviceResults.filter((result) => !result.success).map((result) => <p key={result.serviceId}><strong>{result.serviceId}</strong>{result.errorMessage}</p>)}</div>}
      </aside>
      <div className={`graph-save-blocker ${saving ? 'graph-save-blocker-active' : ''}`} aria-hidden="true" />
    </div>
  );
}

export function GraphWorkspace() {
  return <ReactFlowProvider><GraphWorkspaceInner /></ReactFlowProvider>;
}
