import { FileDiff } from 'lucide-react';

import type { AgentToolCall, JsonValue } from '../api/contracts';

type JsonRecord = Readonly<Record<string, JsonValue>>;

function record(value: JsonValue | undefined): JsonRecord | null {
  return value !== null && value !== undefined && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord : null;
}

function stringField(value: JsonRecord | null, field: string): string {
  const candidate = value?.[field];
  return typeof candidate === 'string' ? candidate : '';
}

function portName(value: string): string {
  return value.split(':').at(-1) ?? value;
}

function operationLabel(operation: JsonRecord): string {
  const kind = stringField(operation, 'op');
  const node = record(operation.node);
  const edge = record(operation.edge);
  switch (kind) {
    case 'createNode': return `Add ${stringField(node, 'name') || stringField(node, 'nodeId')}`;
    case 'connectEdge': return `${stringField(edge, 'fromNodeId')}.${portName(stringField(edge, 'fromPortId'))} -> ${stringField(edge, 'toNodeId')}.${portName(stringField(edge, 'toPortId'))}`;
    case 'setNodeState': return `Set ${stringField(operation, 'nodeId')}.${stringField(operation, 'field')} = ${JSON.stringify(operation.value)}`;
    case 'refreshInstalledSpec': return `Refresh ${stringField(operation, 'nodeId')} definition`;
    case 'renameNode': return `Rename ${stringField(operation, 'nodeId')} to ${stringField(operation, 'name')}`;
    case 'deleteNode': return `Delete ${stringField(operation, 'nodeId')}`;
    case 'disconnectEdge': return `Remove connection ${stringField(operation, 'edgeId')}`;
    case 'setNodeEnabled': return `${operation.enabled === true ? 'Enable' : 'Disable'} ${stringField(operation, 'nodeId')}`;
    case 'setNodeLayout': return `Move ${stringField(record(operation.layout), 'nodeId')}`;
    case 'bindOperatorService': return `Bind ${stringField(operation, 'nodeId')} to ${stringField(operation, 'serviceId')}`;
    case 'setServiceSpec':
    case 'setOperatorSpec': return `Edit ${stringField(operation, 'nodeId')} definition`;
    case 'insertFragment': return 'Insert graph fragment';
    default: return kind || 'Graph change';
  }
}

export function AgentPatchPreview({ call }: { readonly call: AgentToolCall | undefined }) {
  const patch = record(call?.arguments.patch);
  const operations = Array.isArray(patch?.operations)
    ? patch.operations.map(record).filter((item): item is JsonRecord => item !== null)
    : [];
  if (operations.length === 0) return null;
  const count = (kind: string) => operations.filter((operation) => operation.op === kind).length;
  const parts = [
    [count('createNode'), 'nodes'], [count('connectEdge'), 'connections'],
    [count('setNodeState'), 'settings'], [count('refreshInstalledSpec'), 'definitions'],
  ].filter(([amount]) => Number(amount) > 0).map(([amount, label]) => `${amount} ${amount === 1 ? String(label).slice(0, -1) : label}`);
  const categorized = count('createNode') + count('connectEdge') + count('setNodeState') + count('refreshInstalledSpec');
  const other = operations.length - categorized;
  if (other > 0) parts.push(`${other} other ${other === 1 ? 'change' : 'changes'}`);
  const added = operations.filter((operation) => operation.op === 'createNode')
    .map((operation) => stringField(record(operation.node), 'name')).filter(Boolean);

  return <details className="agent-patch-preview">
    <summary><FileDiff size={15} /><span><strong>{parts.join(' · ') || `${operations.length} changes`}</strong>{added.length > 0 && <small>{added.join(', ')}</small>}</span></summary>
    <ol>{operations.map((operation, index) => <li key={index}>{operationLabel(operation)}</li>)}</ol>
    <details className="agent-patch-raw"><summary>Exact patch</summary><pre>{JSON.stringify(patch, null, 2)}</pre></details>
  </details>;
}
