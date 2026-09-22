import type { GraphNode, GraphPort } from '../api/contracts';

export interface PortRow {
  readonly key: string;
  readonly input?: GraphPort;
  readonly output?: GraphPort;
}

export function visibleNodePorts(node: GraphNode): readonly GraphPort[] {
  return node.ports.filter((port) => {
    if (port.kind === 'exec' || port.kind === 'command') return true;
    if (port.kind === 'data') return port.dataSpec?.showOnNode !== false;
    return port.stateSpec?.showOnNode !== false;
  });
}

function rowKey(port: GraphPort): string {
  if (port.kind === 'state') return `state:${port.runtimeName}`;
  return `${port.kind}:${port.name}`;
}

export function nodePortRows(node: GraphNode): readonly PortRow[] {
  const rows = new Map<string, { key: string; input?: GraphPort; output?: GraphPort }>();
  for (const port of visibleNodePorts(node)) {
    const key = rowKey(port);
    const row = rows.get(key) ?? { key };
    if (port.direction === 'input') row.input = port;
    else row.output = port;
    rows.set(key, row);
  }
  return [...rows.values()];
}
