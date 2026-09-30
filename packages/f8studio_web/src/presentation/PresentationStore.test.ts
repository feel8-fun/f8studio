import { expect, test, vi } from 'vitest';

import { studioLive } from '../api/liveStore';
import { parsePresentationCommand, PresentationStore } from './PresentationStore';

test('restores media, 3D and extension outputs from ordered live snapshots', () => {
  class Socket {
    onclose: (() => void) | null = null;
    close() { this.onclose?.(); }
  }
  vi.stubGlobal('WebSocket', Socket);
  const store = new PresentationStore();
  try {
    store.start();
    studioLive.apply({ type: 'live.snapshot', values: {
      'presentation/three/up': { nodeId: 'three', command: 'viz.three_d.world_up', payload: { worldUp: '+z' }, seq: 2 },
      'presentation/three/scene': { nodeId: 'three', command: 'viz.three_d.set', payload: { people: [], worldUp: '+y' }, seq: 1 },
      'presentation/video': { nodeId: 'video', command: 'viz.video.set', payload: { videoStreamKey: 'video/source' }, seq: 3 },
      'presentation/audio': { nodeId: 'audio', command: 'viz.audio.set', payload: { audioStreamKey: 'audio/source' }, seq: 4 },
      'presentation/tcode': { nodeId: 'tcode', command: 'viz.tcode.snapshot', payload: { model: 'SR6', channels: { L0: 5000 } }, seq: 5 },
    } });
    expect(store.getOutputSnapshot('three')?.payload).toEqual({ people: [], worldUp: '+z' });
    expect(store.getOutputSnapshot('video')?.payload.videoStreamKey).toBe('video/source');
    expect(store.getOutputSnapshot('audio')?.payload.audioStreamKey).toBe('audio/source');
    expect(store.getOutputSnapshot('tcode')?.payload.channels).toEqual({ L0: 5000 });
    studioLive.apply({ type: 'live.snapshot', values: {} });
    expect(store.getOutputsSnapshot().size).toBe(0);
  } finally {
    store.stop();
    vi.unstubAllGlobals();
  }
});

test('tracks presentation outputs in delivery order despite source clock changes', () => {
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
    payload: { videoStreamKey: 'stale/video' },
    updatedAt: 2,
  });
  expect(nodeListener).toHaveBeenCalledTimes(2);
  expect(otherNodeListener).not.toHaveBeenCalled();

  store.applyCommand({ nodeId: 'video-1', command: 'viz.video.detach', payload: {}, tsMs: 30 });
  expect(store.getOutputSnapshot('video-1')).toBeNull();
  expect(nodeListener).toHaveBeenCalledTimes(3);
});

test('exposes Audio Viz configuration and removes it on detach', () => {
  const store = new PresentationStore();
  store.applyCommand({
    nodeId: 'audio-1',
    command: 'viz.audio.set',
    payload: { audioStreamKey: 'f8/svc/capture/nodes/capture/data/audio' },
    tsMs: 1,
  });
  expect(store.getOutputSnapshot('audio-1')).toMatchObject({
    renderer: 'audio',
    payload: { audioStreamKey: 'f8/svc/capture/nodes/capture/data/audio' },
  });
  store.applyCommand({ nodeId: 'audio-1', command: 'viz.audio.detach', payload: {}, tsMs: 2 });
  expect(store.getOutputSnapshot('audio-1')).toBeNull();
});

test('merges TCode model metadata and validates event envelopes', () => {
  const store = new PresentationStore();
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.set_model', payload: { model: 'SR6' }, tsMs: 1 });
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.write', payload: { line: 'L05000' }, tsMs: 1 });

  expect(store.getOutputSnapshot('tcode-1')?.payload).toEqual({ line: 'L05000', model: 'SR6', channels: { L0: 5000 } });
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.write', payload: { line: 'R09999I500' }, tsMs: 2 });
  expect(store.getOutputSnapshot('tcode-1')?.payload.channels).toEqual({ L0: 5000, R0: 9999 });
  store.applyCommand({ nodeId: 'tcode-1', command: 'viz.tcode.reset', payload: {}, tsMs: 4 });
  expect(store.getOutputSnapshot('tcode-1')?.payload).toEqual({ line: '', model: 'SR6', channels: {}, resetVersion: 1 });
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

test('requires both live and event connections to report connected', () => {
  class Socket {
    static instances: Socket[] = [];
    onopen: (() => void) | null = null;
    onclose: (() => void) | null = null;
    constructor(readonly url: string) { Socket.instances.push(this); }
    close() { this.onclose?.(); }
  }
  vi.stubGlobal('WebSocket', Socket);
  const store = new PresentationStore();
  try {
    store.start();
    const live = Socket.instances.find((socket) => socket.url.endsWith('/api/live'))!;
    const events = Socket.instances.find((socket) => socket.url.includes('/api/events'))!;
    events.onopen?.();
    expect(store.getConnectionSnapshot()).toBe(false);
    live.onopen?.();
    expect(store.getConnectionSnapshot()).toBe(true);
    live.close();
    expect(store.getConnectionSnapshot()).toBe(false);
  } finally {
    store.stop();
    vi.unstubAllGlobals();
  }
});
