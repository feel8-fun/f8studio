import { moveNodeOperations, resizeServiceOperations } from './layoutEdits';
import { useEdgesState, useNodesInitialized, useNodesState, useReactFlow, useUpdateNodeInternals, type Connection, type Edge, type FinalConnectionState, type Node, type OnNodeDrag, type ResizeParams } from '@xyflow/react';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ApiError, createCatalogNode } from '../api/client';

import type { GraphEdge, GraphNode, GraphOperation, JsonValue, NodeLayout, OperatorSpec, ProjectRecord, ServiceSpec } from '../api/contracts';
import { connectionError, edgeKindForPort } from './connectionRules';
import { absoluteFlowPosition, COMPACT_SERVICE_WIDTH, compactServiceHeight, constrainOperatorPosition, duplicateFragment, OPERATOR_MIN_HEIGHT, OPERATOR_WIDTH, operatorHeight, projectDocument, reconcileProjectedEdges, reconcileProjectedNodes, SERVICE_MIN_HEIGHT, SERVICE_WIDTH, serviceChildInsetY, STUDIO_SERVICE_CLASS, type StudioFlowNode } from './projection';

import { errorMessage, newId } from './workspaceUtils';
const STUDIO_SERVICE_ID = 'studio';

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

export function useGraphCanvas({ project, busy, setBusy, setError, commit, reloadProject, selectedNodeId, selectedEdgeId, setSelectedNodeId, setSelectedEdgeId }: {
  readonly selectedNodeId: string | null;
  readonly selectedEdgeId: string | null;
  readonly setSelectedNodeId: (id: string | null) => void;
  readonly setSelectedEdgeId: (id: string | null) => void;
  readonly project: ProjectRecord | null;
  readonly busy: boolean;
  readonly setBusy: (value: boolean) => void;
  readonly setError: (value: string | null) => void;
  readonly commit: (operations: readonly GraphOperation[]) => Promise<void>;
  readonly reloadProject: (projectId: string) => Promise<ProjectRecord>;
}) {
  const projectId = project?.projectId ?? null;
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [projectedProjectId, setProjectedProjectId] = useState<string | null>(null);
  const projectedProjectIdRef = useRef<string | null>(null);
  const fittedProjectId = useRef<string | null>(null);
  const connectedStateInputsCache = useRef<ReadonlySet<string>>(new Set());
  const projectedEdgeIds = useRef<{ readonly projectId: string | null; readonly ids: ReadonlySet<string> }>({
    projectId: null,
    ids: new Set(),
  });
  const graphCanvasRef = useRef<HTMLElement>(null);
  const nodesInitialized = useNodesInitialized();
  const { fitView, screenToFlowPosition } = useReactFlow<StudioFlowNode, Edge>();
  const updateNodeInternals = useUpdateNodeInternals();
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

  const addSpec = useCallback(async (spec: ServiceSpec | OperatorSpec) => {
    if (project === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      const nodeId = newId(spec.specKind === 'service' ? 'service' : 'operator');
      const nodesToCreate: GraphNode[] = [];
      let selectedNode: GraphNode;
      if (!('operatorClass' in spec)) {
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
    const touch = 'changedTouches' in event ? event.changedTouches.item(0) : null;
    const pointer = 'clientX' in event ? { x: event.clientX, y: event.clientY }
      : touch === null ? undefined : { x: touch.clientX, y: touch.clientY };
    try {
      void commit(moveNodeOperations(project.document, node, nodes, pointer === undefined ? undefined : screenToFlowPosition(pointer)));
    } catch (reason) {
      restoreProjection();
      setError(errorMessage(reason));
    }
  }, [busy, commit, nodes, project, restoreProjection, screenToFlowPosition]);

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
    try {
      void commitRef.current(resizeServiceOperations(project.document, nodeId, bounds));
    } catch (reason) {
      restoreProjection();
      setError(errorMessage(reason));
    }
  };
  const resizeService = useCallback((nodeId: string, bounds: ResizeParams) => {
    resizeServiceRef.current(nodeId, bounds);
  }, []);
  const setNodeState = useCallback((nodeId: string, field: string, value: JsonValue) => {
    void commitRef.current([{ op: 'setNodeState', nodeId, field, value }]);
  }, []);
  return { nodes, edges, onNodesChange, onEdgesChange, graphCanvasRef, selectedNode, selectedEdge, addSpec, connect, connectionEnded, isValidConnection,
    deleteNodes, deleteEdges, moveNode, duplicateSelection, bindOperatorService, connectedStateInputs,
    replaceEdge, removeEdge, resizeService, setNodeState };
}
