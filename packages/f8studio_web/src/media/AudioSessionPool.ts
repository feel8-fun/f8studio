import { closeAudioSession, createAudioSession } from '../api/client';
import { createRtcPeerConnection, waitForIceGatheringComplete } from './rtc';
import { RtcSessionPool, type RtcSessionTransport } from './RtcSessionPool';
export type { RtcSessionSnapshot as AudioSessionSnapshot, RtcSessionLease as AudioSessionLease } from './RtcSessionPool';
export type AudioSessionTransport = RtcSessionTransport<string>;

const defaultTransport: AudioSessionTransport = {
  createPeer: createRtcPeerConnection,
  waitForIce: waitForIceGatheringComplete,
  createSession: createAudioSession,
  closeSession: (sessionId) => closeAudioSession(sessionId, true),
};

export class AudioSessionPool {
  private readonly pool: RtcSessionPool<string>;
  constructor(transport: AudioSessionTransport = defaultTransport) {
    this.pool = new RtcSessionPool('audio', transport);
  }
  acquire(source: string) {
    const normalized = source.trim();
    if (normalized === '') throw new Error('Audio source must not be empty');
    return this.pool.acquire(normalized, normalized);
  }
  closeAll(): void { this.pool.closeAll(); }
}

export const audioSessionPool = new AudioSessionPool();
window.addEventListener('pagehide', (event) => {
  if (!event.persisted) audioSessionPool.closeAll();
});
