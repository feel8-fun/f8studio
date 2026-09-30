
import { Check, Keyboard, Play, RotateCw, Trash2, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { fetchHotkeys, registerHotkey, unregisterHotkey } from '../api/client';

import type { CommandSpec, GraphEdge, GraphNode, GraphOperation, HotkeyBinding, RuntimeMonitor, RuntimeStateField, StateSpec } from '../api/contracts';

import { StateFieldControl } from './StateFieldControl';
import { useRuntimeNodeState } from './useRuntimeNodeState';

import { SchemaEditor } from './SchemaEditor';

import { errorMessage } from './workspaceUtils';
const STUDIO_SERVICE_ID = 'studio';

function hotkeyEligible(field: StateSpec): boolean {
  if (field.access !== 'rw') return false;
  const control = field.control?.kind ?? '';
  if (control === 'button') return field.valueSchema.type === 'integer' || field.valueSchema.type === 'number';
  return ['select', 'dropdown', 'dropbox', 'combo', 'combobox'].includes(control) ||
    ('enum' in field.valueSchema && (field.valueSchema.enum?.length ?? 0) > 0);
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

export function NodeInspector({
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

export function EdgeInspector({
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
