import { afterEach, expect, test, vi } from 'vitest';

import type { AudioSessionAnswer } from '../api/contracts';
import { AudioSessionPool, type AudioSessionTransport } from './AudioSessionPool';

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

afterEach(() => vi.useRealTimers());

test('shares one audio session and closes it after the final viewer leaves', async () => {
  vi.useFakeTimers();
  const peer = new FakePeer();
  const answer: AudioSessionAnswer = {
    sessionId: 'audio-session-1',
    source: 'f8/audio',
    sdp: 'answer-sdp',
    type: 'answer',
    sampleRate: 48000,
    channels: 2,
    transportPolicy: 'bounded-queue-16',
  };
  const transport: AudioSessionTransport = {
    createPeer: vi.fn(async () => peer as unknown as RTCPeerConnection),
    waitForIce: vi.fn(async () => undefined),
    createSession: vi.fn(async () => answer),
    closeSession: vi.fn(async () => undefined),
  };
  const pool = new AudioSessionPool(transport);

  const first = pool.acquire(' f8/audio ');
  const second = pool.acquire('f8/audio');
  await vi.waitFor(() => expect(transport.createSession).toHaveBeenCalledTimes(1));
  expect(transport.createPeer).toHaveBeenCalledTimes(1);

  first.release();
  await vi.runAllTimersAsync();
  expect(transport.closeSession).not.toHaveBeenCalled();

  second.release();
  await vi.advanceTimersByTimeAsync(1499);
  expect(peer.close).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1);
  expect(peer.close).toHaveBeenCalledTimes(1);
  expect(transport.closeSession).toHaveBeenCalledWith('audio-session-1');
});
