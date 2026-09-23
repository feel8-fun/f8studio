import { expect, test, vi } from 'vitest';

import { parsePresentationCommand, PresentationStore } from './PresentationStore';

test('tracks presentation outputs per node and ignores stale commands', () => {
  const store = new PresentationStore();
  const nodeListener = vi.fn();
  const otherNodeListener = vi.fn();
  store.subscribeNode('video-1', nodeListener);
  store.subscribeNode('other', otherNodeListener);

  store.applyCommand({
    nodeId: 'video-1',
    command: 'viz.video.set',
    payload: { videoStreamKey: 'f8/video' },
    tsMs: 20,
  });
  store.applyCommand({
    nodeId: 'video-1',
    command: 'viz.video.set',
    payload: { videoStreamKey: 'stale/video' },
    tsMs: 10,
  });

  expect(store.getOutputSnapshot('video-1')).toMatchObject({
    renderer: 'video',
    payload: { videoStreamKey: 'f8/video' },
    updatedAt: 20,
  });
  expect(nodeListener).toHaveBeenCalledTimes(1);
  expect(otherNodeListener).not.toHaveBeenCalled();

  store.applyCommand({ nodeId: 'video-1', command: 'viz.video.detach', payload: {}, tsMs: 30 });
  expect(store.getOutputSnapshot('video-1')).toBeNull();
  expect(nodeListener).toHaveBeenCalledTimes(2);
});

test('merges TCode model metadata and validates event envelopes', () => {
  const store = new PresentationStore();
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.set_model', payload: { model: 'SR6' }, tsMs: 1 });
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.write', payload: { line: 'L05000' }, tsMs: 1 });

  expect(store.getOutputSnapshot('tcode-1')?.payload).toEqual({ line: 'L05000', model: 'SR6' });
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.reset', payload: {}, tsMs: 2 });
  expect(store.getOutputSnapshot('tcode-1')?.payload).toEqual({ line: '', model: 'SR6' });
  expect(parsePresentationCommand({
    type: 'presentation.command',
    payload: { nodeId: 'video-1', command: 'viz.video.set', payload: { videoStreamKey: 'f8/video' }, tsMs: 4 },
  })).toMatchObject({ nodeId: 'video-1', tsMs: 4 });
  expect(parsePresentationCommand({
    type: 'presentation.command',
    payload: { nodeId: 'video-1', command: 'viz.video.set', payload: [] },
  })).toBeNull();
});

test('keeps a 3D scene when only its world-up setting changes', () => {
  const store = new PresentationStore();
  store.applyCommand({
    nodeId: 'three-1',
    command: 'viz.three_d.set',
    payload: { tsMs: 1, worldUp: '+y', people: [] },
    tsMs: 1,
  });
  store.applyCommand({
    nodeId: 'three-1',
    command: 'viz.three_d.world_up',
    payload: { worldUp: '+z' },
    tsMs: 2,
  });

  expect(store.getOutputSnapshot('three-1')?.payload).toEqual({ tsMs: 1, worldUp: '+z', people: [] });
});

test('evicts the least recently updated output when the store reaches its limit', () => {
  const store = new PresentationStore();
  for (let index = 0; index < 32; index += 1) {
    store.applyCommand({
      nodeId: `node-${index}`,
      command: 'viz.text.update',
      payload: { value: index },
      tsMs: index,
    });
  }
  store.applyCommand({ nodeId: 'node-0', command: 'viz.text.update', payload: { value: 'recent' }, tsMs: 100 });
  store.applyCommand({ nodeId: 'node-32', command: 'viz.text.update', payload: { value: 32 }, tsMs: 101 });

  expect(store.getOutputSnapshot('node-0')?.payload).toEqual({ value: 'recent' });
  expect(store.getOutputSnapshot('node-1')).toBeNull();
  expect(store.getOutputsSnapshot().size).toBe(32);
});
