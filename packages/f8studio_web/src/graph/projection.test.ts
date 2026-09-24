import { expect, test } from 'vitest';

import type { StudioDocument } from '../api/contracts';
import {
  absoluteFlowPosition,
  COMPACT_SERVICE_WIDTH,
  compactServiceHeight,
  duplicateFragment,
  operatorHeight,
  projectDocument,
  reconcileProjectedEdges,
  reconcileProjectedNodes,
  SERVICE_MIN_HEIGHT,
  SERVICE_WIDTH,
  VIDEO_PREVIEW_HEIGHT,
} from './projection';

const document: StudioDocument = {
  schemaVersion: 'f8studio-document/1',
  projectId: 'project1',
  graphId: 'project1',
  graphRevision: 3,
  layoutRevision: 2,
  nodes: [{
    kind: 'service',
    nodeId: 'engine',
    name: 'Engine',
    serviceId: 'engine',
    serviceClass: 'f8.pyengine',
    spec: { serviceClass: 'f8.pyengine', label: 'Engine', specKind: 'service' },
    ports: [{
      portId: 'data:output:value',
      name: 'value',
      runtimeName: 'value',
      kind: 'data',
      direction: 'output',
    }],
    stateValues: {},
    enabled: true,
  }, {
    kind: 'operator',
    nodeId: 'source',
    name: 'Source',
    serviceId: 'engine',
    serviceClass: 'f8.pyengine',
    operatorClass: 'f8.test.source',
    spec: {
      serviceClass: 'f8.pyengine',
      operatorClass: 'f8.test.source',
      label: 'Source',
      specKind: 'operator',
    },
    ports: [{
      portId: 'data:output:value',
      name: 'value',
      runtimeName: 'value',
      kind: 'data',
      direction: 'output',
    }],
    stateValues: {},
    enabled: true,
  }],
  edges: [{
    edgeId: 'loopback',
    fromNodeId: 'source',
    fromPortId: 'data:output:value',
    toNodeId: 'engine',
    toPortId: 'data:output:value',
    kind: 'data',
    strategy: 'latest',
    queueSize: 16,
    timeoutMs: null,
  }],
  layout: [
    { nodeId: 'engine', x: 125, y: 240, collapsed: false },
    { nodeId: 'source', x: 180, y: 350, collapsed: false },
  ],
};

test('projects services before their nested operators using relative flow positions', () => {
  const projected = projectDocument(document);

  expect(projected.nodes).toHaveLength(2);
  expect(projected.nodes[0]).toMatchObject({
    id: 'engine',
    type: 'studio',
    position: { x: 125, y: 240 },
    style: { width: SERVICE_WIDTH, height: SERVICE_MIN_HEIGHT },
    data: { graphNode: { nodeId: 'engine' }, childCount: 1 },
  });
  expect(projected.nodes[1]).toMatchObject({
    id: 'source',
    parentId: 'engine',
    style: { width: 240, height: 64 },
    position: { x: 55, y: 110 },
  });
  expect(absoluteFlowPosition(projected.nodes[1]!, projected.nodes)).toEqual({ x: 180, y: 350 });
  expect(projected.edges[0]).toMatchObject({ id: 'loopback', source: 'source', target: 'engine', zIndex: 2 });
});

test('uses persisted service dimensions while enforcing canvas minimums', () => {
  const expanded: StudioDocument = {
    ...structuredClone(document),
    layout: document.layout.map((layout) => layout.nodeId === 'engine'
      ? { ...layout, width: 940, height: 510 }
      : layout),
  };
  expect(projectDocument(expanded).nodes[0]?.style).toMatchObject({ width: 940, height: 510 });

  const undersized: StudioDocument = {
    ...structuredClone(document),
    layout: document.layout.map((layout) => layout.nodeId === 'engine'
      ? { ...layout, width: 120, height: 100 }
      : layout),
  };
  expect(projectDocument(undersized).nodes[0]?.style).toMatchObject({
    width: SERVICE_WIDTH,
    height: SERVICE_MIN_HEIGHT,
  });

  const legacyDefault: StudioDocument = {
    ...structuredClone(document),
    layout: document.layout.map((layout) => layout.nodeId === 'engine'
      ? { ...layout, width: 620, height: 320 }
      : layout),
  };
  expect(projectDocument(legacyDefault).nodes[0]?.style).toMatchObject({
    width: SERVICE_WIDTH,
    height: 320,
  });
});

test('collapses services without operator children to their own visible rows', () => {
  const leafDocument: StudioDocument = {
    ...structuredClone(document),
    nodes: document.nodes.filter((node) => node.kind === 'service'),
    edges: [],
    layout: [{ nodeId: 'engine', x: 125, y: 240, width: 620, height: 320, collapsed: false }],
  };
  const service = leafDocument.nodes[0];
  expect(service).toBeDefined();
  expect(projectDocument(leafDocument).nodes[0]?.style).toEqual({
    width: COMPACT_SERVICE_WIDTH,
    height: compactServiceHeight(service!),
  });
});

test('projects Studio operators on the root canvas while retaining their runtime binding', () => {
  const engine = document.nodes[0];
  const source = document.nodes[1];
  if (engine?.kind !== 'service' || source?.kind !== 'operator') throw new Error('Invalid projection fixture');
  const studioService = {
    ...engine,
    nodeId: 'studio',
    name: 'Web Studio Runtime',
    serviceId: 'studio',
    serviceClass: 'f8.pystudio',
    spec: { ...engine.spec, serviceClass: 'f8.pystudio' },
  };
  const video = {
    ...source,
    nodeId: 'video',
    name: 'Video Viz',
    serviceId: 'studio',
    serviceClass: 'f8.pystudio',
    operatorClass: 'f8.viz.video',
    spec: { ...source.spec, serviceClass: 'f8.pystudio', operatorClass: 'f8.viz.video' },
  };
  const studioDocument: StudioDocument = {
    ...document,
    nodes: [...document.nodes, studioService, video],
    edges: [],
    layout: [
      ...document.layout,
      { nodeId: 'studio', x: 700, y: 100, width: 850, height: 600, collapsed: false },
      { nodeId: 'video', x: 30, y: 940, collapsed: false },
    ],
  };
  const projected = projectDocument(studioDocument);
  const runtime = projected.nodes.find((node) => node.id === 'studio');
  const viz = projected.nodes.find((node) => node.id === 'video');

  expect(runtime).toMatchObject({
    data: { childCount: 0 },
    style: { width: COMPACT_SERVICE_WIDTH, height: 64 },
  });
  expect(viz).toMatchObject({ position: { x: 30, y: 940 }, data: { graphNode: { serviceId: 'studio' } } });
  expect(viz?.parentId).toBeUndefined();
  expect(projected.nodes.find((node) => node.id === 'source')?.parentId).toBe('engine');
  expect(absoluteFlowPosition(viz!, projected.nodes)).toEqual({ x: 30, y: 940 });

  const reloaded = projectDocument(structuredClone(studioDocument));
  expect(reloaded.nodes.find((node) => node.id === 'video')?.position).toEqual({ x: 30, y: 940 });
});

test('reuses unchanged projected graph objects after a server round trip', () => {
  const current = projectDocument(document);
  const roundTripped = projectDocument(structuredClone(document));
  const nodes = reconcileProjectedNodes(current.nodes, roundTripped.nodes);
  const edges = reconcileProjectedEdges(current.edges, roundTripped.edges);

  expect(nodes).toBe(current.nodes);
  expect(edges).toBe(current.edges);
  expect(nodes[0]).toBe(current.nodes[0]);
  expect(nodes[1]).toBe(current.nodes[1]);
  expect(edges[0]).toBe(current.edges[0]);
});

test('preserves measured dimensions while replacing changed projected node data', () => {
  const current = projectDocument(document);
  current.nodes[1] = {
    ...current.nodes[1]!,
    measured: { width: 240, height: 64 },
    selected: true,
  };
  const changedDocument: StudioDocument = {
    ...structuredClone(document),
    nodes: document.nodes.map((node) => node.nodeId === 'source' ? { ...node, name: 'Renamed Source' } : node),
  };
  const projected = projectDocument(changedDocument);
  const nodes = reconcileProjectedNodes(current.nodes, projected.nodes);

  expect(nodes).not.toBe(current.nodes);
  expect(nodes[0]).toBe(current.nodes[0]);
  expect(nodes[1]).not.toBe(current.nodes[1]);
  expect(nodes[1]).toMatchObject({
    data: { graphNode: { name: 'Renamed Source' } },
    measured: { width: 240, height: 64 },
    selected: true,
  });
});

test('sizes compact operators from their fixed port-row geometry', () => {
  const operator = document.nodes.find((node) => node.kind === 'operator');
  expect(operator).toBeDefined();
  expect(operatorHeight(operator!)).toBe(64);
  expect(operatorHeight({
    ...operator!,
    ports: [
      ...operator!.ports,
      { portId: 'state:input:a', name: 'a', runtimeName: 'a', kind: 'state', direction: 'input' },
      { portId: 'state:input:b', name: 'b', runtimeName: 'b', kind: 'state', direction: 'input' },
    ],
  })).toBe(112);

  expect(operatorHeight({
    ...operator!,
    operatorClass: 'f8.viz.video',
    spec: {
      ...operator!.spec,
      operatorClass: 'f8.viz.video',
      rendererClass: 'viz_video',
    },
  })).toBe(64 + VIDEO_PREVIEW_HEIGHT);
});

test('duplicates a service with its operators, internal edges, and absolute layout', () => {
  let sequence = 0;
  const operation = duplicateFragment(document, new Set(['engine']), (prefix) => `${prefix}_copy_${sequence += 1}`);

  expect(operation).not.toBeNull();
  expect(operation?.nodes).toHaveLength(2);
  expect(operation?.nodes[0]).toMatchObject({
    kind: 'service',
    nodeId: 'service_copy_1',
    serviceId: 'service_copy_1',
    name: 'Engine Copy',
  });
  expect(operation?.nodes[1]).toMatchObject({
    kind: 'operator',
    nodeId: 'operator_copy_2',
    serviceId: 'service_copy_1',
    name: 'Source Copy',
  });
  expect(operation?.edges[0]).toMatchObject({
    edgeId: 'edge_copy_3',
    fromNodeId: 'operator_copy_2',
    toNodeId: 'service_copy_1',
  });
  expect(operation?.layout[0]).toMatchObject({
    nodeId: 'service_copy_1', x: 165, y: 280, width: SERVICE_WIDTH, height: SERVICE_MIN_HEIGHT,
  });
  expect(operation?.layout[1]).toMatchObject({ nodeId: 'operator_copy_2', x: 220, y: 390 });
});
