import { afterEach, expect, test, vi } from 'vitest';

import type { MediaSessionAnswer } from '../api/contracts';
import { VideoSessionPool, type VideoQuality, type VideoSessionTransport } from './VideoSessionPool';

class FakePeer {
  localDescription: RTCSessionDescription | null = null;
  connectionState: RTCPeerConnectionState = 'new';
  onconnectionstatechange: ((event: Event) => void) | null = null;
  ontrack: ((event: RTCTrackEvent) => void) | null = null;
  readonly close = vi.fn(() => {
    this.connectionState = 'closed';
  });
  readonly addTransceiver = vi.fn(() => ({} as RTCRtpTransceiver));
  readonly createOffer = vi.fn(async (): Promise<RTCSessionDescriptionInit> => ({ type: 'offer', sdp: 'offer-sdp' }));
  readonly setLocalDescription = vi.fn(async (description: RTCLocalSessionDescriptionInit) => {
    this.localDescription = description as RTCSessionDescription;
  });
  readonly setRemoteDescription = vi.fn(async (_description: RTCSessionDescriptionInit) => undefined);
}

afterEach(() => {
  vi.useRealTimers();
});

test('shares one negotiation and releases it after the final consumer', async () => {
  vi.useFakeTimers();
  const peer = new FakePeer();
  const answer: MediaSessionAnswer = {
    sessionId: 'session-1',
    source: 'f8/video',
    quality: 'thumbnail',
    sdp: 'answer-sdp',
    type: 'answer',
    maxWidth: 640,
    maxHeight: 360,
    maxFps: 30,
    overlay: false,
  };
  const transport: VideoSessionTransport = {
    createPeer: vi.fn(async () => peer as unknown as RTCPeerConnection),
    waitForIce: vi.fn(async () => undefined),
    createSession: vi.fn(async () => answer),
    closeSession: vi.fn(async () => undefined),
  };
  const pool = new VideoSessionPool(transport);

  const first = pool.acquire(' f8/video ', 'thumbnail');
  const second = pool.acquire('f8/video', 'thumbnail');
  await vi.waitFor(() => expect(transport.createSession).toHaveBeenCalledTimes(1));
  expect(transport.createPeer).toHaveBeenCalledTimes(1);

  first.release();
  await vi.runAllTimersAsync();
  expect(transport.closeSession).not.toHaveBeenCalled();

  second.release();
  await vi.runAllTimersAsync();
  expect(peer.close).toHaveBeenCalledTimes(1);
  expect(transport.closeSession).toHaveBeenCalledWith('session-1');
});

test('cancels deferred shutdown when a replacement consumer mounts', async () => {
  vi.useFakeTimers();
  const peer = new FakePeer();
  const answer: MediaSessionAnswer = {
    sessionId: 'session-1', source: 'f8/video', quality: 'thumbnail', sdp: 'answer-sdp', type: 'answer',
    maxWidth: 640, maxHeight: 360, maxFps: 30, overlay: false,
  };
  const transport: VideoSessionTransport = {
    createPeer: vi.fn(async () => peer as unknown as RTCPeerConnection),
    waitForIce: vi.fn(async () => undefined),
    createSession: vi.fn(async () => answer),
    closeSession: vi.fn(async () => undefined),
  };
  const pool = new VideoSessionPool(transport);

  const first = pool.acquire('f8/video', 'thumbnail');
  await vi.waitFor(() => expect(transport.createSession).toHaveBeenCalledTimes(1));
  first.release();
  const replacement = pool.acquire('f8/video', 'thumbnail');
  await vi.runAllTimersAsync();

  expect(transport.createPeer).toHaveBeenCalledTimes(1);
  expect(transport.closeSession).not.toHaveBeenCalled();
  replacement.release();
  await vi.runAllTimersAsync();
  expect(transport.closeSession).toHaveBeenCalledTimes(1);
});

test('closes all retained sessions when the owning page exits', async () => {
  const firstPeer = new FakePeer();
  const secondPeer = new FakePeer();
  const peers = [firstPeer, secondPeer];
  const transport: VideoSessionTransport = {
    createPeer: vi.fn(async () => peers.shift() as unknown as RTCPeerConnection),
    waitForIce: vi.fn(async () => undefined),
    createSession: vi.fn(async (source: string, quality: VideoQuality): Promise<MediaSessionAnswer> => ({
      sessionId: `session-${source}`,
      source,
      quality,
      sdp: 'answer-sdp',
      type: 'answer',
      maxWidth: 640,
      maxHeight: 360,
      maxFps: 30,
      overlay: false,
    })),
    closeSession: vi.fn(async () => undefined),
  };
  const pool = new VideoSessionPool(transport);
  pool.acquire('f8/video-a', 'thumbnail');
  pool.acquire('f8/video-b', 'thumbnail');
  await vi.waitFor(() => expect(transport.createSession).toHaveBeenCalledTimes(2));

  pool.closeAll();
  await vi.waitFor(() => expect(transport.closeSession).toHaveBeenCalledTimes(2));
  expect(firstPeer.close).toHaveBeenCalledTimes(1);
  expect(secondPeer.close).toHaveBeenCalledTimes(1);
});
