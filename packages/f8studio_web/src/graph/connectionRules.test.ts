import { expect, test } from 'vitest';

import type { GraphNode, GraphPort, StudioDocument } from '../api/contracts';
import { connectionError } from './connectionRules';

function port(
  portId: string,
  kind: GraphPort['kind'],
  direction: GraphPort['direction'],
  access: 'rw' | 'ro' | 'wo' = 'rw',
): GraphPort {
  return {
    portId,
    name: portId,
    runtimeName: portId,
    kind,
    direction,
    dataSpec: kind === 'data' ? { name: portId, valueSchema: { type: 'number' }, payloadKind: 'json' } : null,
    stateSpec: kind === 'state' ? { name: portId, valueSchema: { type: 'number' }, access } : null,
  };
}

function operator(nodeId: string, serviceId: string, ports: readonly GraphPort[]): GraphNode {
  return {
    kind: 'operator',
    nodeId,
    name: nodeId,
    serviceId,
    serviceClass: 'f8.pyengine',
    operatorClass: `test.${nodeId}`,
    spec: {
      serviceClass: 'f8.pyengine', operatorClass: `test.${nodeId}`, label: nodeId, specKind: 'operator',
    },
    ports,
    stateValues: {},
    enabled: true,
  };
}

function document(nodes: readonly GraphNode[]): StudioDocument {
  return {
    schemaVersion: 'f8studio-document/2', projectId: 'project', graphId: 'graph',
    graphRevision: 0, layoutRevision: 0, nodes, edges: [], layout: [],
  };
}

test('accepts compatible data connections and rejects occupied inputs', () => {
  const source = operator('source', 'service', [port('out', 'data', 'output')]);
  const sink = operator('sink', 'service', [port('in', 'data', 'input')]);
  const base = document([source, sink]);
  const connection = { source: 'source', sourceHandle: 'out', target: 'sink', targetHandle: 'in' };

  expect(connectionError(base, connection)).toBeNull();
  expect(connectionError({
    ...base,
    edges: [{
      edgeId: 'edge', fromNodeId: 'source', fromPortId: 'out', toNodeId: 'sink', toPortId: 'in',
      kind: 'data', strategy: 'latest', queueSize: 16, timeoutMs: null,
    }],
  }, connection)).toContain('already connected');
});

test('enforces exec service ownership and state access/cycle rules', () => {
  const first = operator('first', 'service-a', [
    port('exec-out', 'exec', 'output'),
    { ...port('state-out', 'state', 'output'), runtimeName: 'value' },
    { ...port('state-in', 'state', 'input'), runtimeName: 'value' },
  ]);
  const second = operator('second', 'service-b', [
    port('exec-in', 'exec', 'input'),
    { ...port('state-out', 'state', 'output'), runtimeName: 'value' },
    { ...port('state-in', 'state', 'input'), runtimeName: 'value' },
  ]);
  const base = document([first, second]);

  expect(connectionError(base, {
    source: 'first', sourceHandle: 'exec-out', target: 'second', targetHandle: 'exec-in',
  })).toContain('cannot cross service');
  const withStateEdge: StudioDocument = {
    ...base,
    edges: [{
      edgeId: 'state-edge', fromNodeId: 'first', fromPortId: 'state-out',
      toNodeId: 'second', toPortId: 'state-in', kind: 'state', strategy: 'latest', queueSize: 16, timeoutMs: null,
    }],
  };
  expect(connectionError(withStateEdge, {
    source: 'second', sourceHandle: 'state-out', target: 'first', targetHandle: 'state-in',
  })).toContain('create a cycle');
});
