import { afterEach, expect, test, vi } from 'vitest';
import { RtcSessionPool, type RtcSessionTransport } from './RtcSessionPool';

function setup() {
  const peer = {
    localDescription: { type: 'offer', sdp: 'offer' },
    close: vi.fn(), addTransceiver: vi.fn(),
    createOffer: vi.fn(async () => ({ type: 'offer', sdp: 'offer' })),
    setLocalDescription: vi.fn(async () => {}),
    setRemoteDescription: vi.fn(async () => {}),
  };
  const transport: RtcSessionTransport<string> = {
    createPeer: vi.fn(async () => peer as unknown as RTCPeerConnection),
    waitForIce: vi.fn(async () => {}),
    createSession: vi.fn(async () => ({ sessionId: 'session', type: 'answer' as const, sdp: 'answer' })),
    closeSession: vi.fn(async () => {}),
  };
  return { peer, transport, pool: new RtcSessionPool('audio', transport) };
}

afterEach(() => vi.useRealTimers());

test('negotiation survives a temporary absence of viewers during the grace period', async () => {
  vi.useFakeTimers();
  const { pool, peer, transport } = setup();
  const first = pool.acquire('source', 'source');
  first.release();
  await vi.advanceTimersByTimeAsync(500);
  expect(peer.setRemoteDescription).toHaveBeenCalledTimes(1);
  expect(peer.close).not.toHaveBeenCalled();
  const replacement = pool.acquire('source', 'source');
  await vi.advanceTimersByTimeAsync(1500);
  expect(transport.createSession).toHaveBeenCalledTimes(1);
  expect(peer.close).not.toHaveBeenCalled();
  replacement.release();
  await vi.advanceTimersByTimeAsync(1500);
  expect(transport.closeSession).toHaveBeenCalledTimes(1);
});

test('releasing a lease invalidated by closeAll cannot evict its replacement', async () => {
  vi.useFakeTimers();
  const { pool, transport } = setup();
  const stale = pool.acquire('source', 'source');
  await vi.advanceTimersByTimeAsync(1);
  pool.closeAll();
  const replacement = pool.acquire('source', 'source');
  stale.release();
  await vi.advanceTimersByTimeAsync(1500);
  const another = pool.acquire('source', 'source');
  await vi.advanceTimersByTimeAsync(1);
  expect(transport.createPeer).toHaveBeenCalledTimes(2);
  replacement.release(); another.release();
  await vi.advanceTimersByTimeAsync(1500);
  expect(transport.closeSession).toHaveBeenCalledTimes(2);
});

test('a server answer arriving after closeAll is reclaimed', async () => {
  vi.useFakeTimers();
  const { pool, transport, peer } = setup();
  let resolve!: (value: { sessionId: string; type: 'answer'; sdp: string }) => void;
  const answer = new Promise<{ sessionId: string; type: 'answer'; sdp: string }>((accept) => { resolve = accept; });
  vi.mocked(transport.createSession).mockReturnValue(answer);
  pool.acquire('source', 'source');
  await vi.advanceTimersByTimeAsync(1);
  pool.closeAll();
  resolve({ sessionId: 'late', type: 'answer', sdp: 'answer' });
  await vi.advanceTimersByTimeAsync(1);
  expect(transport.closeSession).toHaveBeenCalledExactlyOnceWith('late');
  expect(peer.setRemoteDescription).not.toHaveBeenCalled();
});
