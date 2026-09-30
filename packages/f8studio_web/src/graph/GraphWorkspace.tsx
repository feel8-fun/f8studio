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
  useUpdateNodeInternals,
  type Connection,
  type Edge,
  type FinalConnectionState,
  type Node,
  type OnNodeDrag,
  type ResizeParams,
} from '@xyflow/react';
import { Bot, Check, Copy, Download, Keyboard, Play, Plus, Redo2, RotateCcw, RotateCw, Square, Trash2, Upload, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';

import {
  ApiError,
  changeHistory,
  createCatalogNode,
  createProject,
  deleteProject,
  deployProject,
  exportProjectGraph,
  fetchCatalog,
  fetchDeployJob,
  fetchLatestDeployment,
  fetchHotkeys,
  fetchProject,
  fetchProjects,
  importProjectGraph,
  patchProject,
  refreshCatalog,
  registerHotkey,
  restartProjectService,
  stopProject,
  unregisterHotkey,
} from '../api/client';
import { isStudioDocument, isDeployJob } from '../api/contracts';
import { studioEvents } from '../api/eventStream';
import { useLivePrefix } from '../api/liveStore';
import type {
  CatalogSnapshot,
  CommandSpec,
  DeployJob,
  GraphEdge,
  GraphNode,
  GraphOperation,
  HotkeyBinding,
  JsonValue,
  NodeLayout,
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
  STUDIO_SERVICE_CLASS,
  type StudioFlowNode,
} from './projection';
import { StateFieldControl } from './StateFieldControl';
import { useRuntimeNodeState } from './useRuntimeNodeState';
import { GraphNodeInteractionContext, StudioNodeView } from './StudioNodeView';
import { CommandDialog } from './CommandDialog';
import { SchemaEditor } from './SchemaEditor';
import { MutationQueue } from './MutationQueue';
import { NodeCatalog } from './NodeCatalog';
import { commandResultDetail, runCommand } from './runCommand';

const nodeTypes = { studio: StudioNodeView };
const SELECTED_PROJECT_KEY = 'f8studio.selectedProjectId';
const INSPECTOR_WIDTH_KEY = 'f8studio.graphInspectorWidth';
const INSPECTOR_MIN_WIDTH = 280;
const INSPECTOR_MAX_WIDTH = 640;
const INSPECTOR_DEFAULT_WIDTH = 360;
const STUDIO_SERVICE_ID = 'studio';

function savedInspectorWidth(): number {
  const stored = localStorage.getItem(INSPECTOR_WIDTH_KEY);
  const value = stored === null ? INSPECTOR_DEFAULT_WIDTH : Number(stored);
  return Number.isFinite(value) ? Math.max(INSPECTOR_MIN_WIDTH, Math.min(INSPECTOR_MAX_WIDTH, value)) : INSPECTOR_DEFAULT_WIDTH;
}

function visibleServicePosition(
  center: { readonly x: number; readonly y: number },
  bounds: { readonly left: number; readonly top: number; readonly right: number; readonly bottom: number },
  width: number,
  height: number,
  existing: readonly StudioFlowNode[],
): { readonly x: number; readonly y: number } {
  const margin = 24;
  const minX = bounds.left + margin;
  const minY = bounds.top + margin;
  const maxX = bounds.right - width - margin;
  const maxY = bounds.bottom - height - margin;
  const clampX = (x: number) => maxX < minX ? center.x - width / 2 : Math.max(minX, Math.min(x, maxX));
  const clampY = (y: number) => maxY < minY ? center.y - height / 2 : Math.max(minY, Math.min(y, maxY));
  const offsets: readonly (readonly [number, number])[] = [
    [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, 1], [1, -1], [-1, -1],
  ];
  const positions = offsets.map(([column, row]) => ({
    x: clampX(center.x - width / 2 + column * (width + margin)),
    y: clampY(center.y - height / 2 + row * (height + margin)),
  }));
  return positions.find((position) => existing.every((node) => {
    const nodePosition = absoluteFlowPosition(node, existing);
    const nodeWidth = typeof node.style?.width === 'number' ? node.style.width : OPERATOR_WIDTH;
    const nodeHeight = typeof node.style?.height === 'number' ? node.style.height : OPERATOR_MIN_HEIGHT;
    return position.x + width + margin <= nodePosition.x || nodePosition.x + nodeWidth + margin <= position.x ||
      position.y + height + margin <= nodePosition.y || nodePosition.y + nodeHeight + margin <= position.y;
  })) ?? { x: clampX(center.x - width / 2), y: clampY(center.y - height / 2) };
}

function hotkeyEligible(field: StateSpec): boolean {
  if (field.access !== 'rw') return false;
  const control = field.control?.kind ?? '';
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

function NodeInspector({
  projectId,
  node,
  services,
  monitor,
  busy,
  pendingCommands,
  commit,
  bindService,
  connectedStateInputs,
  onCommand,
  onRestartService,
}: {
  readonly projectId: string;
  readonly node: GraphNode;
  readonly services: readonly GraphNode[];
  readonly monitor: RuntimeMonitor | null;
  readonly busy: boolean;
  readonly pendingCommands: ReadonlySet<string>;
  readonly commit: (operations: readonly GraphOperation[]) => Promise<void>;
  readonly bindService: (nodeId: string, serviceId: string) => void;
  readonly connectedStateInputs: ReadonlySet<string>;
  readonly onCommand: (node: GraphNode, command: CommandSpec) => void;
  readonly onRestartService: (serviceId: string) => void;
}) {
  const fields = node.spec.stateFields ?? [];
  const runtimeFieldNames = useMemo(
    () => fields.filter((field) => field.name !== 'svcId' && field.name !== 'operatorId' && field.access !== 'wo')
      .map((field) => field.name),
    [fields],
  );
  const runtimeValues = useRuntimeNodeState(node, runtimeFieldNames);

  const runtimeValue = (field: StateSpec): RuntimeStateField | undefined => {
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
    {node.kind === 'service' && node.serviceId !== STUDIO_SERVICE_ID && <button type="button" className="command-button" disabled={busy}
      onClick={() => onRestartService(node.serviceId)} title="Restart service, refresh node catalog, and redeploy project">
      <RotateCw size={14} /> Restart service
    </button>}
    {monitor === null ? <p className="monitor-empty">No monitor sample</p> : <dl className="monitor-values">
      <dt>Status</dt><dd>{monitor.alive ? (monitor.ready ? 'Ready' : 'Starting') : 'Offline'}</dd>
      <dt>CPU</dt><dd>{(monitor.cpu?.processPercent ?? 0).toFixed(1)}%</dd>
      <dt>Memory</dt><dd>{((monitor.memory?.rssBytes ?? 0) / 1048576).toFixed(1)} MiB</dd>
      <dt>Queue</dt><dd>{monitor.queue?.depth ?? 0}</dd>
      <dt>Latency p95</dt><dd>{(monitor.timing?.latencyMsP95 ?? 0).toFixed(1)} ms</dd>
    </dl>}
    {fields.length > 0 && <h2>State values</h2>}
    <div className="inspector-fields">{fields.map((field) => {
      const connected = connectedStateInputs.has(`${node.nodeId}:${field.name}`);
      return <div className="inspector-state-field" key={field.name}>
        <StateFieldControl
          node={node}
          field={field}
          disabled={busy}
          connected={connected}
          runtimeValue={runtimeValue(field)}
          runtimeValues={runtimeValues}
          projectId={projectId}
          onCommit={(value) => void commit([{ op: 'setNodeState', nodeId: node.nodeId, field: field.name, value }])}
        />
        {hotkeyEligible(field) && <HotkeyEditor projectId={projectId} node={node} field={field} disabled={busy || connected} />}
      </div>;
    })}</div>
    {(node.spec.commands ?? []).length > 0 && <><h2>Commands</h2><div className="inspector-commands">
      {(node.spec.commands ?? []).map((command) => <button key={command.name} type="button" className="command-button" disabled={busy || pendingCommands.has(`${node.nodeId}:${command.name}`)}
        title={command.description} onClick={() => onCommand(node, command)}><Play size={13} />{command.name}</button>)}
    </div></>}
    <SchemaEditor node={node} busy={busy} commit={commit} />
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

function GraphWorkspaceInner({ onShowOutput }: { readonly onShowOutput: (nodeId: string) => void }) {
  const [inspectorWidth, setInspectorWidth] = useState(savedInspectorWidth);
  const inspectorWidthRef = useRef(inspectorWidth);
  const inspectorResizingRef = useRef(false);
  const graphWorkspaceRef = useRef<HTMLDivElement>(null);
  const [projects, setProjects] = useState<readonly ProjectSummary[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [catalog, setCatalog] = useState<CatalogSnapshot | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [activeCommand, setActiveCommand] = useState<{ readonly node: GraphNode; readonly command: CommandSpec } | null>(null);
  const [commandToast, setCommandToast] = useState<{ readonly id: number; readonly kind: 'success' | 'error'; readonly title: string; readonly detail: string } | null>(null);
  const [pendingCommands, setPendingCommands] = useState<ReadonlySet<string>>(new Set());
  const pendingCommandsRef = useRef(new Set<string>());
  const [busy, setBusy] = useState(false);
  const [refreshingCatalog, setRefreshingCatalog] = useState(false);
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
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [deployment, setDeployment] = useState<DeployJob | null>(null);
  const monitorValues = useLivePrefix('monitor/');
  const monitors = useMemo(() => [...monitorValues.values()] as unknown as readonly RuntimeMonitor[], [monitorValues]);
  const [projectedProjectId, setProjectedProjectId] = useState<string | null>(null);
  const projectedProjectIdRef = useRef<string | null>(null);
  const fittedProjectId = useRef<string | null>(null);
  const mutationInFlight = useRef(false);
  const [mutations] = useState(() => new MutationQueue());
  const connectedStateInputsCache = useRef<ReadonlySet<string>>(new Set());
  const projectedEdgeIds = useRef<{ readonly projectId: string | null; readonly ids: ReadonlySet<string> }>({
    projectId: null,
    ids: new Set(),
  });
  const projectRef = useRef<ProjectRecord | null>(null);
  const graphCanvasRef = useRef<HTMLElement>(null);
  const graphImportRef = useRef<HTMLInputElement>(null);
  const nodesInitialized = useNodesInitialized();
  const { fitView, screenToFlowPosition } = useReactFlow<StudioFlowNode, Edge>();
  const updateNodeInternals = useUpdateNodeInternals();
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
          setSelectedNodeId(null);
          setSelectedEdgeId(null);
          setActiveCommand(null);
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
  }, [projectId, reloadProject]);


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
    if (projectId === null || projectedProjectId !== projectId) return;
    const previous = projectedEdgeIds.current;
    const added = previous.projectId === projectId
      ? edges.filter((edge) => !previous.ids.has(edge.id))
      : [];
    projectedEdgeIds.current = { projectId, ids: new Set(edges.map((edge) => edge.id)) };
    if (added.length === 0) return;
    // React Flow can retain pre-connection handle bounds until the connected nodes are measured again.
    const affectedNodeIds = [...new Set(added.flatMap((edge) => [edge.source, edge.target]))];
    const frame = requestAnimationFrame(() => updateNodeInternals(affectedNodeIds));
    return () => cancelAnimationFrame(frame);
  }, [edges, projectId, projectedProjectId, updateNodeInternals]);

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

  const mutateProject = useCallback((targetId: string, operation: (current: ProjectRecord) => Promise<ProjectRecord>): Promise<void> => {
    mutationInFlight.current = true;
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
      mutationInFlight.current = mutations.pending > 0;
      setSaving(mutationInFlight.current);
    }, (reason: unknown) => {
      mutationInFlight.current = mutations.pending > 0;
      setSaving(mutationInFlight.current);
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
    const isStudioRuntime = operator.data.graphNode.serviceClass === STUDIO_SERVICE_CLASS;
    const width = typeof service.style?.width === 'number' ? service.style.width : SERVICE_WIDTH;
    const height = typeof service.style?.height === 'number' ? service.style.height : SERVICE_MIN_HEIGHT;
    const relative = isStudioRuntime ? null : constrainOperatorPosition(
      operator.position,
      width,
      height,
      operatorHeight(operator.data.graphNode),
      serviceChildInsetY(service.data.graphNode),
    );
    const position = relative === null ? operator.position : {
      x: service.position.x + relative.x,
      y: service.position.y + relative.y,
    };
    const currentLayout = project.document.layout.find((layout) => layout.nodeId === nodeId);
    void commit([
      { op: 'bindOperatorService', nodeId, serviceId },
      {
        op: 'setNodeLayout',
        layout: {
          nodeId,
          x: position.x,
          y: position.y,
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
    setSelectedProjectId(projectId);
    setProject(null);
    setDeployment(null);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setActiveCommand(null);
    try {
      await reloadProject(projectId);
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
      setSelectedProjectId(created.projectId);
      setProject(created);
      setDeployment(null);
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setActiveCommand(null);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [projects.length]);

  const removeProject = useCallback(async () => {
    const selected = projects.find((item) => item.projectId === selectedProjectId);
    if (selected === undefined || busy || saving || mutationInFlight.current) return;
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
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setActiveCommand(null);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, projects, saving, selectProject, selectedProjectId]);

  const openAgent = useCallback(() => {
    if (project === null) return;
    const url = new URL(window.location.href);
    url.search = new URLSearchParams({ view: 'agent', project: project.projectId }).toString();
    const popup = window.open('', `f8_agent_${encodeURIComponent(project.projectId)}`, 'popup,width=1000,height=760');
    if (popup === null) {
      window.alert('Allow pop-ups for Studio to open the Agent window.');
      return;
    }
    try {
      const current = new URL(popup.location.href);
      if (current.searchParams.get('view') !== 'agent' || current.searchParams.get('project') !== project.projectId) {
        popup.location.assign(url.toString());
      }
    } catch (reason) {
      console.error('Cannot inspect the existing Agent window', reason);
      popup.location.assign(url.toString());
    }
    popup.focus();
  }, [project]);

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
        const compatibleServices = project.document.nodes.filter(
          (node) => node.kind === 'service' && node.serviceClass === spec.serviceClass,
        );
        const currentSelection = project.document.nodes.find((node) => node.nodeId === selectedNodeId);
        let service = compatibleServices.find((node) => node.nodeId === currentSelection?.nodeId ||
          node.serviceId === (currentSelection?.kind === 'operator' ? currentSelection.serviceId : null));
        if (service === undefined && compatibleServices.length === 1) service = compatibleServices[0];
        if (service === undefined && compatibleServices.length > 1) {
          throw new Error(`Select the target ${spec.serviceClass} service before adding this operator`);
        }
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
      const flowElement = graphCanvasRef.current?.querySelector('.react-flow');
      const flowBounds = flowElement?.getBoundingClientRect();
      if (flowBounds === undefined || flowBounds.width <= 0 || flowBounds.height <= 0) {
        throw new Error('Graph viewport is not ready');
      }
      const topLeft = screenToFlowPosition({ x: flowBounds.left, y: flowBounds.top });
      const bottomRight = screenToFlowPosition({ x: flowBounds.right, y: flowBounds.bottom });
      const visible = {
        left: topLeft.x, top: topLeft.y, right: bottomRight.x, bottom: bottomRight.y,
      };
      const center = screenToFlowPosition({
        x: flowBounds.left + flowBounds.width / 2,
        y: flowBounds.top + flowBounds.height / 2,
      });
      const existing = projectDocument(project.document).nodes;
      const serviceLayouts = new Map<string, NodeLayout>();
      for (const node of nodesToCreate) {
        if (node.kind !== 'service') continue;
        const width = COMPACT_SERVICE_WIDTH;
        const height = compactServiceHeight(node);
        const position = visibleServicePosition(center, visible, width, height, existing);
        serviceLayouts.set(node.nodeId, {
          nodeId: node.nodeId, x: position.x, y: position.y, width, height, collapsed: false,
        });
        existing.push({
          id: node.nodeId, type: 'studio', position, style: { width, height },
          data: { graphNode: node, childCount: 0 },
        });
      }
      const studioOperatorPosition = selectedNode.kind === 'operator' && selectedNode.serviceClass === STUDIO_SERVICE_CLASS
        ? projectDocument({ ...project.document, nodes: [...project.document.nodes, ...nodesToCreate],
            layout: [...project.document.layout, ...serviceLayouts.values()] })
          .nodes.find((node) => node.id === selectedNode.nodeId)?.position
        : undefined;
      const operations = nodesToCreate.map((node): GraphOperation => {
        if (node.kind !== 'service') return {
          op: 'createNode',
          node,
          ...(studioOperatorPosition === undefined ? {} : {
            layout: {
              nodeId: node.nodeId,
              x: studioOperatorPosition.x,
              y: studioOperatorPosition.y,
              collapsed: false,
            },
          }),
        };
        const layout = serviceLayouts.get(node.nodeId);
        if (layout === undefined) throw new Error(`Missing layout for service ${node.nodeId}`);
        return {
          op: 'createNode',
          node,
          layout,
        };
      });
      await commit(operations);
      setSelectedNodeId(selectedNode.nodeId);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 409) await reloadProject(project.projectId);
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, commit, project, reloadProject, screenToFlowPosition, selectedNodeId]);

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
        candidate.nodeId === graphNode.nodeId || (graphNode.serviceClass !== STUDIO_SERVICE_CLASS &&
          candidate.kind === 'operator' && candidate.serviceId === graphNode.serviceId)
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
      if (graphNode.serviceClass === STUDIO_SERVICE_CLASS) {
        for (const candidate of projected.nodes) {
          if (candidate.data.graphNode.kind !== 'operator' ||
            candidate.data.graphNode.serviceId !== graphNode.serviceId ||
            project.document.layout.some((layout) => layout.nodeId === candidate.id)) continue;
          const position = absoluteFlowPosition(candidate, projected.nodes);
          operations.push({
            op: 'setNodeLayout',
            layout: { nodeId: candidate.id, x: position.x, y: position.y, collapsed: false },
          });
        }
      }
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
      if (candidate.data.graphNode.kind !== 'service' ||
        candidate.data.graphNode.serviceClass === STUDIO_SERVICE_CLASS) return false;
      const width = typeof candidate.style?.width === 'number' ? candidate.style.width : SERVICE_WIDTH;
      const height = typeof candidate.style?.height === 'number' ? candidate.style.height : SERVICE_MIN_HEIGHT;
      return dropPosition.x >= candidate.position.x && dropPosition.x <= candidate.position.x + width &&
        dropPosition.y >= candidate.position.y && dropPosition.y <= candidate.position.y + height;
    });
    if (target === undefined) {
      if (graphNode.serviceClass !== STUDIO_SERVICE_CLASS) {
        restoreProjection();
        setError('Operators must remain inside a compatible service container.');
        return;
      }
      const currentLayout = project.document.layout.find((layout) => layout.nodeId === node.id);
      void commit([{
        op: 'setNodeLayout',
        layout: {
          nodeId: node.id,
          x: absolute.x,
          y: absolute.y,
          width: currentLayout?.width,
          height: currentLayout?.height,
          collapsed: currentLayout?.collapsed ?? false,
        },
      }]);
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
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }, [busy, mutateProject, project]);

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

  const refreshNodeCatalog = useCallback(async () => {
    if (busy || refreshingCatalog) return;
    setRefreshingCatalog(true);
    setError(null);
    try {
      const updated = await refreshCatalog();
      setCatalog(updated);
      setCommandToast({ id: Date.now(), kind: 'success', title: 'Node catalog refreshed', detail: `${updated.operators.length} operators available` });
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setRefreshingCatalog(false);
    }
  }, [busy, refreshingCatalog]);

  const restartService = useCallback(async (serviceId: string) => {
    if (project === null || busy || refreshingCatalog) return;
    setBusy(true);
    setError(null);
    try {
      const initialJob = await restartProjectService(project.projectId, serviceId);
      setCatalog(await fetchCatalog());
      await followDeployment(initialJob);
      setCommandToast({ id: Date.now(), kind: 'success', title: 'Service restarted', detail: `${serviceId} restarted, node catalog refreshed, and project deployed` });
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
  useEffect(() => {
    if (commandToast === null) return;
    const timer = window.setTimeout(() => setCommandToast((current) => current?.id === commandToast.id ? null : current),
      commandToast.kind === 'error' ? 8000 : 5000);
    return () => window.clearTimeout(timer);
  }, [commandToast]);
  const reportCommand = useCallback((kind: 'success' | 'error', title: string, detail: string) => {
    setCommandToast({ id: Date.now(), kind, title, detail });
  }, []);
  const openCommand = useCallback((node: GraphNode, command: CommandSpec) => {
    if ((command.params ?? []).length > 0) {
      setActiveCommand({ node, command });
      return;
    }
    const key = `${node.nodeId}:${command.name}`;
    if (pendingCommandsRef.current.has(key)) return;
    pendingCommandsRef.current.add(key);
    setPendingCommands(new Set(pendingCommandsRef.current));
    void runCommand(node, command, {}).then((response) => {
      reportCommand('success', `${node.name}: ${command.name}`, commandResultDetail(node, response));
    }, (reason: unknown) => {
      reportCommand('error', `${node.name}: ${command.name} failed`, errorMessage(reason));
    }).finally(() => {
      pendingCommandsRef.current.delete(key);
      setPendingCommands(new Set(pendingCommandsRef.current));
    });
  }, [reportCommand]);
  const nodeInteraction = useMemo(() => ({
    busy,
    pendingCommands,
    connectedStateInputs,
    resizeService,
    setState: setNodeState,
    openCommand,
    showOutput: onShowOutput,
  }), [busy, pendingCommands, connectedStateInputs, resizeService, setNodeState, openCommand, onShowOutput]);
  const selectedMonitor = selectedNode === null ? null : monitors.find((monitor) => monitor.nodeId === selectedNode.nodeId) ??
    (selectedNode.kind === 'service' ? monitors.find((monitor) => monitor.serviceId === selectedNode.serviceId) ?? null : null);
  const locked = busy || saving;

  const resizeInspector = (clientX: number): void => {
    const bounds = graphWorkspaceRef.current?.getBoundingClientRect();
    if (bounds === undefined) return;
    const availableMax = Math.max(INSPECTOR_MIN_WIDTH, bounds.width - 238 - 280 - 6);
    const next = Math.max(INSPECTOR_MIN_WIDTH, Math.min(INSPECTOR_MAX_WIDTH, availableMax, bounds.right - clientX - 3));
    inspectorWidthRef.current = next;
    setInspectorWidth(next);
  };
  const finishInspectorResize = (): void => {
    if (!inspectorResizingRef.current) return;
    inspectorResizingRef.current = false;
    localStorage.setItem(INSPECTOR_WIDTH_KEY, String(inspectorWidthRef.current));
  };

  return (
    <div className="graph-workspace" ref={graphWorkspaceRef} style={{ '--inspector-width': `${inspectorWidth}px` } as CSSProperties}>
      <aside className="graph-palette" aria-label="Node catalog">
        <div className="project-control">
          <label htmlFor="project-select">Project</label>
          <div>
            <select id="project-select" value={selectedProjectId ?? ''} disabled={locked} onChange={(event) => void selectProject(event.target.value)}>
              <option value="" disabled>Select project</option>
              {projects.map((item) => <option key={item.projectId} value={item.projectId}>{item.name}</option>)}
            </select>
            <button type="button" className="small-icon-button" title="New project" aria-label="New project" disabled={locked} onClick={() => void addProject()}><Plus size={16} /></button>
            <button type="button" className="small-icon-button" title="Delete project" aria-label="Delete project" disabled={locked || selectedProjectId === null} onClick={() => void removeProject()}><Trash2 size={15} /></button>
          </div>
        </div>
        <NodeCatalog catalog={catalog} projectServiceClasses={new Set(project?.document.nodes.filter((node) => node.kind === 'service').map((node) => node.serviceClass))}
          canAdd={!busy && !refreshingCatalog && project !== null} refreshing={busy || refreshingCatalog}
          onAdd={(spec) => void addSpec(spec)} onRefresh={() => void refreshNodeCatalog()} />
      </aside>

      <section className="graph-canvas" aria-label="Graph canvas" ref={graphCanvasRef}>
        <div className="graph-toolbar">
          <button type="button" title="Open Agent window" aria-label="Open Agent window" disabled={project === null} onClick={openAgent}><Bot size={16} /></button>
          <button type="button" title="Undo" aria-label="Undo" disabled={busy || project === null} onClick={() => void history('undo')}><RotateCcw size={16} /></button>
          <button type="button" title="Redo" aria-label="Redo" disabled={busy || project === null} onClick={() => void history('redo')}><Redo2 size={16} /></button>
          <button type="button" title="Duplicate selection" aria-label="Duplicate selection" disabled={busy || project === null || (selectedNodeId === null && !nodes.some((node) => node.selected))} onClick={duplicateSelection}><Copy size={15} /></button>
          <button type="button" title="Export graph" aria-label="Export graph" disabled={busy || project === null} onClick={() => void downloadGraph()}><Download size={15} /></button>
          <button type="button" title="Import graph" aria-label="Import graph" disabled={busy || project === null} onClick={() => graphImportRef.current?.click()}><Upload size={15} /></button>
          <input ref={graphImportRef} type="file" accept=".json,application/json" hidden aria-label="Graph import file" onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = '';
            if (file !== undefined) void uploadGraph(file);
          }} />
          <button type="button" title="Deploy" aria-label="Deploy" disabled={busy || project === null} onClick={() => void deploy()}><Play size={16} /></button>
          <button type="button" title="Stop services" aria-label="Stop services" disabled={stopping || project === null} onClick={() => void stop()}><Square size={14} /></button>
          <span>{project === null ? 'No project selected' : `Draft r${project.document.graphRevision} · Layout r${project.document.layoutRevision}`}</span>
          <span className={`deploy-state deploy-${deployment?.status ?? 'none'}`}>{deployment === null ? 'Not deployed' : `${deployment.status} r${deployment.sourceGraphRevision}`}</span>
          <span className="save-state">{saving ? 'Saving...' : busy ? 'Working...' : 'Saved'}</span>
        </div>
        {project === null ? <div className="empty-state"><p>{selectedProjectId === null ? 'Create a project to start building a graph.' : 'This project could not be loaded.'}</p>{selectedProjectId === null && <button className="command-button primary" type="button" onClick={() => void addProject()}>New project</button>}</div> :
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

      <div className="graph-inspector-resizer" role="separator" aria-label="Resize Inspector" aria-orientation="vertical"
        aria-valuemin={INSPECTOR_MIN_WIDTH} aria-valuemax={INSPECTOR_MAX_WIDTH} aria-valuenow={inspectorWidth} tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          inspectorResizingRef.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => { if (inspectorResizingRef.current) resizeInspector(event.clientX); }}
        onPointerUp={(event) => {
          finishInspectorResize();
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={finishInspectorResize}
        onLostPointerCapture={finishInspectorResize}
        onKeyDown={(event) => {
          const bounds = graphWorkspaceRef.current?.getBoundingClientRect();
          const maximum = bounds === undefined ? INSPECTOR_MAX_WIDTH : Math.min(INSPECTOR_MAX_WIDTH, Math.max(INSPECTOR_MIN_WIDTH, bounds.width - 238 - 280 - 6));
          let next = inspectorWidth;
          if (event.key === 'ArrowLeft') next += 20;
          else if (event.key === 'ArrowRight') next -= 20;
          else if (event.key === 'Home') next = INSPECTOR_MIN_WIDTH;
          else if (event.key === 'End') next = maximum;
          else return;
          event.preventDefault();
          event.stopPropagation();
          next = Math.max(INSPECTOR_MIN_WIDTH, Math.min(maximum, next));
          inspectorWidthRef.current = next;
          setInspectorWidth(next);
          localStorage.setItem(INSPECTOR_WIDTH_KEY, String(next));
        }} />
      <aside className="graph-inspector" aria-label="Inspector">
        <h2>Inspector</h2>
        {selectedNode !== null && project !== null ? <NodeInspector projectId={project.projectId} node={selectedNode} services={project.document.nodes} monitor={selectedMonitor} busy={busy} pendingCommands={pendingCommands} commit={commit} bindService={bindOperatorService} connectedStateInputs={connectedStateInputs} onCommand={openCommand} onRestartService={(serviceId) => void restartService(serviceId)} /> :
          selectedEdge !== null ? <EdgeInspector edge={selectedEdge} nodes={project?.document.nodes ?? []} busy={busy} replace={replaceEdge} remove={removeEdge} /> :
            <p>Select a node or connection to inspect it.</p>}
        {deployment !== null && deployment.serviceResults.some((result) => !result.success) && <div className="deploy-errors">{deployment.serviceResults.filter((result) => !result.success).map((result) => <p key={result.serviceId}><strong>{result.serviceId}</strong>{result.errorMessage}</p>)}</div>}
      </aside>
      {activeCommand !== null && <CommandDialog key={`${activeCommand.node.nodeId}:${activeCommand.command.name}`} node={activeCommand.node} command={activeCommand.command} onClose={() => setActiveCommand(null)} onResult={reportCommand} />}
      {commandToast !== null && <div className={`command-toast command-toast-${commandToast.kind}`} role={commandToast.kind === 'error' ? 'alert' : 'status'}>
        <div><strong>{commandToast.title}</strong><button type="button" className="icon-button" title="Dismiss" aria-label="Dismiss command result" onClick={() => setCommandToast(null)}><X size={14} /></button></div>
        <p>{commandToast.detail}</p>
      </div>}
      <div className={`graph-save-blocker ${saving ? 'graph-save-blocker-active' : ''}`} aria-hidden="true" />
    </div>
  );
}

export function GraphWorkspace({ onShowOutput }: { readonly onShowOutput: (nodeId: string) => void }) {
  return <ReactFlowProvider><GraphWorkspaceInner onShowOutput={onShowOutput} /></ReactFlowProvider>;
}
