import type { GraphEdgeKind, GraphPort, StudioDocument } from '../api/contracts';

export interface GraphConnection {
  readonly source: string | null;
  readonly sourceHandle: string | null;
  readonly target: string | null;
  readonly targetHandle: string | null;
}

export function edgeKindForPort(port: GraphPort): GraphEdgeKind {
  if (port.kind === 'data') return 'data';
  if (port.kind === 'exec') return 'exec';
  return 'state';
}

function dataPayloadKind(port: GraphPort): string {
  return port.dataSpec?.payloadKind ?? port.dataSpec?.payload?.kind ?? 'json';
}

function stateKey(nodeId: string, port: GraphPort): string {
  return `${nodeId}\u0000${port.runtimeName}`;
}

function createsStateCycle(
  document: StudioDocument,
  sourceNodeId: string,
  sourcePort: GraphPort,
  targetNodeId: string,
  targetPort: GraphPort,
): boolean {
  const nodes = new Map(document.nodes.map((node) => [node.nodeId, node]));
  const adjacency = new Map<string, string[]>();
  const add = (source: string, target: string) => {
    const downstream = adjacency.get(source) ?? [];
    downstream.push(target);
    adjacency.set(source, downstream);
  };
  for (const edge of document.edges) {
    if (edge.kind !== 'state') continue;
    const edgeSourceNode = nodes.get(edge.fromNodeId);
    const edgeTargetNode = nodes.get(edge.toNodeId);
    const edgeSourcePort = edgeSourceNode?.ports.find((port) => port.portId === edge.fromPortId);
    const edgeTargetPort = edgeTargetNode?.ports.find((port) => port.portId === edge.toPortId);
    if (edgeSourcePort !== undefined && edgeTargetPort !== undefined) {
      add(stateKey(edge.fromNodeId, edgeSourcePort), stateKey(edge.toNodeId, edgeTargetPort));
    }
  }
  const source = stateKey(sourceNodeId, sourcePort);
  const target = stateKey(targetNodeId, targetPort);
  add(source, target);
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (key: string): boolean => {
    if (visiting.has(key)) return true;
    if (visited.has(key)) return false;
    visiting.add(key);
    for (const downstream of adjacency.get(key) ?? []) {
      if (visit(downstream)) return true;
    }
    visiting.delete(key);
    visited.add(key);
    return false;
  };
  return [...adjacency.keys()].some(visit);
}

export function connectionError(document: StudioDocument, connection: GraphConnection): string | null {
  if (connection.source === null || connection.target === null ||
    connection.sourceHandle === null || connection.targetHandle === null) {
    return 'A connection requires source and target ports.';
  }
  const sourceNode = document.nodes.find((node) => node.nodeId === connection.source);
  const targetNode = document.nodes.find((node) => node.nodeId === connection.target);
  const sourcePort = sourceNode?.ports.find((port) => port.portId === connection.sourceHandle);
  const targetPort = targetNode?.ports.find((port) => port.portId === connection.targetHandle);
  if (sourceNode === undefined || targetNode === undefined || sourcePort === undefined || targetPort === undefined) {
    return 'The selected node or port is no longer available.';
  }
  if (sourcePort.direction !== 'output' || targetPort.direction !== 'input') {
    return 'Connections must run from an output port to an input port.';
  }
  const kind = edgeKindForPort(sourcePort);
  if (kind !== edgeKindForPort(targetPort)) return 'Only compatible port types can be connected.';
  if (document.edges.some((edge) => edge.fromNodeId === sourceNode.nodeId && edge.fromPortId === sourcePort.portId &&
    edge.toNodeId === targetNode.nodeId && edge.toPortId === targetPort.portId)) {
    return 'These ports are already connected.';
  }
  if (document.edges.some((edge) => edge.toNodeId === targetNode.nodeId && edge.toPortId === targetPort.portId)) {
    return `${targetNode.name}.${targetPort.name} already has an upstream connection.`;
  }
  if (kind === 'data') {
    if (dataPayloadKind(sourcePort) !== dataPayloadKind(targetPort)) {
      return `Data payload mismatch: ${dataPayloadKind(sourcePort)} cannot connect to ${dataPayloadKind(targetPort)}.`;
    }
    return null;
  }
  if (kind === 'exec') {
    if (sourceNode.kind !== 'operator' || targetNode.kind !== 'operator') {
      return 'Exec connections require operator endpoints.';
    }
    if (sourceNode.serviceId !== targetNode.serviceId) return 'Exec connections cannot cross service boundaries.';
    if (document.edges.some((edge) => edge.fromNodeId === sourceNode.nodeId && edge.fromPortId === sourcePort.portId)) {
      return `${sourceNode.name}.${sourcePort.name} already has a downstream exec connection.`;
    }
    return null;
  }
  if (sourcePort.stateSpec?.access === 'wo') return `${sourceNode.name}.${sourcePort.name} is write-only.`;
  if (targetPort.stateSpec?.access === 'ro') return `${targetNode.name}.${targetPort.name} is read-only.`;
  if (createsStateCycle(document, sourceNode.nodeId, sourcePort, targetNode.nodeId, targetPort)) {
    return 'This state connection would create a cycle.';
  }
  return null;
}
