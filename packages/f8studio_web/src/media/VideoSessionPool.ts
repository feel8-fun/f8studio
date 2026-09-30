import { closeMediaSession, createMediaSession } from '../api/client';
import { createRtcPeerConnection, waitForIceGatheringComplete } from './rtc';

export type VideoQuality = 'thumbnail' | 'main';

import { RtcSessionPool } from './RtcSessionPool';
export type { RtcSessionSnapshot as VideoSessionSnapshot, RtcSessionLease as VideoSessionLease } from './RtcSessionPool';

export interface VideoSessionTransport {
  readonly createPeer: () => Promise<RTCPeerConnection>;
  readonly waitForIce: (peer: RTCPeerConnection) => Promise<void>;
  readonly createSession: (source: string, quality: VideoQuality, description: RTCSessionDescriptionInit) => ReturnType<typeof createMediaSession>;
  readonly closeSession: (sessionId: string) => Promise<void>;
}

const defaultTransport: VideoSessionTransport = {
  createPeer: createRtcPeerConnection,
  waitForIce: waitForIceGatheringComplete,
  createSession: (source, quality, description) => createMediaSession(source, quality, description),
  closeSession: (sessionId) => closeMediaSession(sessionId, true),
};

interface VideoSource { readonly source: string; readonly quality: VideoQuality }

export class VideoSessionPool {
  private readonly pool: RtcSessionPool<VideoSource>;
  constructor(transport: VideoSessionTransport = defaultTransport) {
    this.pool = new RtcSessionPool('video', {
      createPeer: transport.createPeer,
      waitForIce: transport.waitForIce,
      createSession: (source, description) => transport.createSession(source.source, source.quality, description),
      closeSession: transport.closeSession,
    });
  }
  acquire(source: string, quality: VideoQuality) {
    const normalized = source.trim();
    if (normalized === '') throw new Error('Video source must not be empty');
    return this.pool.acquire(`${quality}:${normalized}`, { source: normalized, quality });
  }
  closeAll(): void { this.pool.closeAll(); }
}

export const videoSessionPool = new VideoSessionPool();
window.addEventListener('pagehide', (event) => {
  if (!event.persisted) videoSessionPool.closeAll();
});
