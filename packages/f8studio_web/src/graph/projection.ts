import { MarkerType, type Edge, type Node, type XYPosition } from '@xyflow/react';

import type { GraphNode, GraphOperation, NodeLayout, StudioDocument } from '../api/contracts';
import { nodePortRows } from './portRows';

export const SERVICE_WIDTH = 620;
export const SERVICE_MIN_HEIGHT = 320;
export const OPERATOR_WIDTH = 260;
export const OPERATOR_MIN_HEIGHT = 80;
export const CONTAINER_INSET_X = 32;
export const CONTAINER_INSET_Y = 96;
export const OPERATOR_GAP_X = 24;
export const OPERATOR_GAP_Y = 28;

export interface StudioNodeData extends Record<string, unknown> {
  readonly graphNode: GraphNode;
  readonly childCount: number;
}

export type StudioFlowNode = Node<StudioNodeData, 'studio'>;

function jsonStructureEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
    return left.every((item, index) => jsonStructureEqual(item, right[index]));
  }
  if (typeof left !== 'object' || left === null || typeof right !== 'object' || right === null) return false;
  const leftRecord = left as Readonly<Record<string, unknown>>;
  const rightRecord = right as Readonly<Record<string, unknown>>;
  const leftKeys = Object.keys(leftRecord);
  if (leftKeys.length !== Object.keys(rightRecord).length) return false;
  return leftKeys.every((key) => Object.hasOwn(rightRecord, key) &&
    jsonStructureEqual(leftRecord[key], rightRecord[key]));
}

function projectedNodeEqual(left: StudioFlowNode, right: StudioFlowNode): boolean {
  return left.id === right.id &&
    left.type === right.type &&
    left.className === right.className &&
    left.dragHandle === right.dragHandle &&
    left.parentId === right.parentId &&
    left.position.x === right.position.x &&
    left.position.y === right.position.y &&
    left.style?.width === right.style?.width &&
    left.style?.height === right.style?.height &&
    left.zIndex === right.zIndex &&
    left.data.childCount === right.data.childCount &&
    jsonStructureEqual(left.data.graphNode, right.data.graphNode);
}

function projectedEdgeEqual(left: Edge, right: Edge): boolean {
  return left.id === right.id &&
    left.source === right.source &&
    left.sourceHandle === right.sourceHandle &&
    left.target === right.target &&
    left.targetHandle === right.targetHandle &&
    left.type === right.type &&
    left.markerEnd === right.markerEnd &&
    left.className === right.className &&
    left.zIndex === right.zIndex;
}

export function reconcileProjectedNodes(
  current: StudioFlowNode[],
  projected: readonly StudioFlowNode[],
): StudioFlowNode[] {
  const currentById = new Map(current.map((node) => [node.id, node]));
  const reconciled = projected.map((node) => {
    const existing = currentById.get(node.id);
    if (existing === undefined) return node;
    if (projectedNodeEqual(existing, node)) return existing;
    return {
      ...node,
      measured: existing.measured,
      selected: existing.selected,
    };
  });
  return current.length === reconciled.length && reconciled.every((node, index) => node === current[index])
    ? current
    : reconciled;
}

export function reconcileProjectedEdges(current: Edge[], projected: readonly Edge[]): Edge[] {
  const currentById = new Map(current.map((edge) => [edge.id, edge]));
  const reconciled = projected.map((edge) => {
    const existing = currentById.get(edge.id);
    if (existing === undefined) return edge;
    if (projectedEdgeEqual(existing, edge)) return existing;
    return { ...edge, selected: existing.selected };
  });
  return current.length === reconciled.length && reconciled.every((edge, index) => edge === current[index])
    ? current
    : reconciled;
}

export function operatorHeight(node: GraphNode): number {
  return Math.max(OPERATOR_MIN_HEIGHT, 52 + Math.max(1, nodePortRows(node).length) * 28);
}

function operatorPositions(children: readonly GraphNode[]): ReadonlyMap<string, XYPosition> {
  const positions = new Map<string, XYPosition>();
  let y = CONTAINER_INSET_Y;
  for (let index = 0; index < children.length; index += 2) {
    const left = children[index];
    const right = children[index + 1];
    if (left !== undefined) positions.set(left.nodeId, { x: CONTAINER_INSET_X, y });
    if (right !== undefined) positions.set(right.nodeId, { x: CONTAINER_INSET_X + OPERATOR_WIDTH + OPERATOR_GAP_X, y });
    y += Math.max(left === undefined ? 0 : operatorHeight(left), right === undefined ? 0 : operatorHeight(right)) + OPERATOR_GAP_Y;
  }
  return positions;
}

function serviceSize(layout: NodeLayout | undefined, children: readonly GraphNode[]): { width: number; height: number } {
  const positions = operatorPositions(children);
  const contentBottom = children.reduce((bottom, child) => {
    const position = positions.get(child.nodeId);
    return position === undefined ? bottom : Math.max(bottom, position.y + operatorHeight(child));
  }, CONTAINER_INSET_Y);
  return {
    width: Math.max(layout?.width ?? SERVICE_WIDTH, SERVICE_WIDTH),
    height: Math.max(layout?.height ?? SERVICE_MIN_HEIGHT, SERVICE_MIN_HEIGHT, contentBottom + 32),
  };
}

function serviceDefaultPosition(index: number): XYPosition {
  return {
    x: 80 + (index % 2) * (SERVICE_WIDTH + 80),
    y: 80 + Math.floor(index / 2) * (SERVICE_MIN_HEIGHT + 80),
  };
}

export function constrainOperatorPosition(
  position: XYPosition,
  width: number,
  height: number,
  childHeight = OPERATOR_MIN_HEIGHT,
): XYPosition {
  return {
    x: Math.max(CONTAINER_INSET_X, Math.min(position.x, width - OPERATOR_WIDTH - CONTAINER_INSET_X)),
    y: Math.max(CONTAINER_INSET_Y, Math.min(position.y, height - childHeight - 32)),
  };
}

export function absoluteFlowPosition(
  node: Pick<StudioFlowNode, 'id' | 'parentId' | 'position'>,
  nodes: readonly Pick<StudioFlowNode, 'id' | 'parentId' | 'position'>[],
): XYPosition {
  if (node.parentId === undefined) return node.position;
  const parent = nodes.find((candidate) => candidate.id === node.parentId);
  if (parent === undefined) throw new Error(`Missing parent flow node ${node.parentId} for ${node.id}`);
  const parentPosition = absoluteFlowPosition(parent, nodes);
  return { x: parentPosition.x + node.position.x, y: parentPosition.y + node.position.y };
}

export function projectDocument(document: StudioDocument): {
  readonly nodes: StudioFlowNode[];
  readonly edges: Edge[];
} {
  const layouts = new Map(document.layout.map((layout) => [layout.nodeId, layout]));
  const services = document.nodes.filter((node) => node.kind === 'service');
  const operators = document.nodes.filter((node) => node.kind === 'operator');
  const operatorsByServiceId = new Map<string, GraphNode[]>();
  for (const operator of operators) {
    const children = operatorsByServiceId.get(operator.serviceId);
    if (children === undefined) operatorsByServiceId.set(operator.serviceId, [operator]);
    else children.push(operator);
  }
  const serviceNodes: StudioFlowNode[] = services.map((service, serviceIndex) => {
    const layout = layouts.get(service.nodeId);
    const position = layout === undefined ? serviceDefaultPosition(serviceIndex) : { x: layout.x, y: layout.y };
    const children = operatorsByServiceId.get(service.serviceId) ?? [];
    const size = serviceSize(layout, children);
    return {
      id: service.nodeId,
      type: 'studio',
      className: 'flow-node-service',
      dragHandle: '.node-drag-handle',
      position,
      style: size,
      data: { graphNode: service, childCount: children.length },
      zIndex: 0,
    };
  });
  const serviceFlowNodes = new Map(serviceNodes.map((node) => [node.data.graphNode.serviceId, node]));
  const defaultPositions = new Map<string, XYPosition>();
  for (const service of services) {
    const children = operatorsByServiceId.get(service.serviceId) ?? [];
    for (const [nodeId, position] of operatorPositions(children)) defaultPositions.set(nodeId, position);
  }
  const operatorNodes: StudioFlowNode[] = operators.map((operator) => {
    const parent = serviceFlowNodes.get(operator.serviceId);
    if (parent === undefined) throw new Error(`Missing service container ${operator.serviceId} for ${operator.nodeId}`);
    const layout = layouts.get(operator.nodeId);
    const defaultPosition = defaultPositions.get(operator.nodeId) ?? { x: CONTAINER_INSET_X, y: CONTAINER_INSET_Y };
    const absolute = layout === undefined
      ? {
          x: parent.position.x + defaultPosition.x,
          y: parent.position.y + defaultPosition.y,
        }
      : { x: layout.x, y: layout.y };
    const width = typeof parent.style?.width === 'number' ? parent.style.width : SERVICE_WIDTH;
    const height = typeof parent.style?.height === 'number' ? parent.style.height : SERVICE_MIN_HEIGHT;
    return {
      id: operator.nodeId,
      type: 'studio',
      className: 'flow-node-operator',
      dragHandle: '.node-drag-handle',
      parentId: parent.id,
      style: { width: OPERATOR_WIDTH, height: operatorHeight(operator) },
      position: constrainOperatorPosition(
        { x: absolute.x - parent.position.x, y: absolute.y - parent.position.y },
        width,
        height,
        operatorHeight(operator),
      ),
      data: { graphNode: operator, childCount: 0 },
      zIndex: 1,
    };
  });
  return {
    nodes: [...serviceNodes, ...operatorNodes],
    edges: document.edges.map((edge) => ({
      id: edge.edgeId,
      source: edge.fromNodeId,
      sourceHandle: edge.fromPortId,
      target: edge.toNodeId,
      targetHandle: edge.toPortId,
      type: 'smoothstep',
      markerEnd: MarkerType.ArrowClosed,
      className: `graph-edge graph-edge-${edge.kind}`,
      zIndex: 2,
    })),
  };
}

export function duplicateFragment(
  document: StudioDocument,
  selectedNodeIds: ReadonlySet<string>,
  makeId: (prefix: string) => string,
): Extract<GraphOperation, { readonly op: 'insertFragment' }> | null {
  const expandedNodeIds = new Set(selectedNodeIds);
  const selectedServiceIds = new Set(document.nodes.flatMap((node) =>
    node.kind === 'service' && selectedNodeIds.has(node.nodeId) ? [node.serviceId] : [],
  ));
  for (const node of document.nodes) {
    if (node.kind === 'operator' && selectedServiceIds.has(node.serviceId)) expandedNodeIds.add(node.nodeId);
  }
  const selectedNodes = document.nodes.filter((node) => expandedNodeIds.has(node.nodeId));
  if (selectedNodes.length === 0) return null;
  const nodeIds = new Map(selectedNodes.map((node) => [node.nodeId, makeId(node.kind)]));
  const projected = projectDocument(document);
  const absolutePositions = new Map(projected.nodes.map((node) => [
    node.id,
    absoluteFlowPosition(node, projected.nodes),
  ]));
  const nodes: GraphNode[] = selectedNodes.map((node) => {
    const nodeId = nodeIds.get(node.nodeId);
    if (nodeId === undefined) throw new Error(`Missing duplicated node id for ${node.nodeId}`);
    if (node.kind === 'service') return { ...node, nodeId, serviceId: nodeId, name: `${node.name} Copy` };
    return {
      ...node,
      nodeId,
      serviceId: nodeIds.get(node.serviceId) ?? node.serviceId,
      name: `${node.name} Copy`,
    };
  });
  const edges = document.edges.flatMap((edge) => {
    const fromNodeId = nodeIds.get(edge.fromNodeId);
    const toNodeId = nodeIds.get(edge.toNodeId);
    if (fromNodeId === undefined || toNodeId === undefined) return [];
    return [{ ...edge, edgeId: makeId('edge'), fromNodeId, toNodeId }];
  });
  const layout = selectedNodes.map((node) => {
    const nodeId = nodeIds.get(node.nodeId);
    const position = absolutePositions.get(node.nodeId);
    if (nodeId === undefined || position === undefined) throw new Error(`Missing duplicated layout for ${node.nodeId}`);
    const currentLayout = document.layout.find((item) => item.nodeId === node.nodeId);
    return {
      nodeId,
      x: position.x + 40,
      y: position.y + 40,
      width: node.kind === 'service' ? currentLayout?.width ?? SERVICE_WIDTH : currentLayout?.width,
      height: node.kind === 'service' ? currentLayout?.height ?? SERVICE_MIN_HEIGHT : currentLayout?.height,
      collapsed: false,
    };
  });
  return { op: 'insertFragment', nodes, edges, layout };
}
