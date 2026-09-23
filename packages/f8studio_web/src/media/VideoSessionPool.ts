import { closeMediaSession, createMediaSession } from '../api/client';
import { createRtcPeerConnection, waitForIceGatheringComplete } from './rtc';

export type VideoQuality = 'thumbnail' | 'main';

export type VideoSessionSnapshot =
  | { readonly kind: 'connecting'; readonly stream: MediaStream | null }
  | { readonly kind: 'playing'; readonly stream: MediaStream }
  | { readonly kind: 'error'; readonly stream: null; readonly message: string };

type Listener = () => void;

export interface VideoSessionLease {
  readonly getSnapshot: () => VideoSessionSnapshot;
  readonly subscribe: (listener: Listener) => () => void;
  readonly retry: () => void;
  readonly release: () => void;
}

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

class PooledVideoSession {
  private readonly listeners = new Set<Listener>();
  private snapshot: VideoSessionSnapshot = { kind: 'connecting', stream: null };
  private peer: RTCPeerConnection | null = null;
  private sessionId: string | null = null;
  private generation = 0;
  private references = 0;
  private started = false;

  constructor(
    readonly source: string,
    readonly quality: VideoQuality,
    private readonly transport: VideoSessionTransport,
  ) {}

  retain(): void {
    this.references += 1;
    if (!this.started) {
      this.started = true;
      void this.connect();
    }
  }

  release(): number {
    this.references = Math.max(0, this.references - 1);
    return this.references;
  }

  referenceCount(): number {
    return this.references;
  }

  getSnapshot = (): VideoSessionSnapshot => this.snapshot;

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  retry = (): void => {
    if (this.references === 0) return;
    this.disconnect();
    this.started = true;
    void this.connect();
  };

  close(): void {
    this.references = 0;
    this.disconnect();
    this.listeners.clear();
  }

  private setSnapshot(snapshot: VideoSessionSnapshot): void {
    this.snapshot = snapshot;
    for (const listener of this.listeners) listener();
  }

  private async connect(): Promise<void> {
    const generation = ++this.generation;
    this.setSnapshot({ kind: 'connecting', stream: null });
    let peer: RTCPeerConnection | null = null;
    let createdSessionId: string | null = null;
    try {
      peer = await this.transport.createPeer();
      if (!this.isCurrent(generation)) {
        peer.close();
        return;
      }
      const activePeer = peer;
      this.peer = activePeer;
      activePeer.addTransceiver('video', { direction: 'recvonly' });
      activePeer.ontrack = (event) => {
        if (!this.isCurrent(generation) || this.peer !== activePeer) return;
        const stream = event.streams[0] ?? new MediaStream([event.track]);
        this.setSnapshot({ kind: 'playing', stream });
      };
      activePeer.onconnectionstatechange = () => {
        if (!this.isCurrent(generation) || this.peer !== activePeer) return;
        if (activePeer.connectionState === 'failed' || activePeer.connectionState === 'closed') {
          this.setSnapshot({ kind: 'error', stream: null, message: `WebRTC ${activePeer.connectionState}` });
        }
      };
      const offer = await activePeer.createOffer();
      await activePeer.setLocalDescription(offer);
      await this.transport.waitForIce(activePeer);
      if (!this.isCurrent(generation)) {
        activePeer.close();
        return;
      }
      if (activePeer.localDescription === null) throw new Error('Browser did not create a local description');
      const answer = await this.transport.createSession(this.source, this.quality, activePeer.localDescription);
      createdSessionId = answer.sessionId;
      if (!this.isCurrent(generation)) {
        activePeer.close();
        await this.releaseServerSession(createdSessionId, 'stale video negotiation');
        return;
      }
      this.sessionId = createdSessionId;
      await activePeer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    } catch (reason: unknown) {
      peer?.close();
      const serverSessionId = createdSessionId !== null && this.sessionId === createdSessionId
        ? createdSessionId
        : null;
      if (serverSessionId !== null) {
        this.sessionId = null;
        await this.releaseServerSession(serverSessionId, 'failed video negotiation');
      }
      if (this.isCurrent(generation)) {
        this.peer = null;
        this.setSnapshot({
          kind: 'error',
          stream: null,
          message: reason instanceof Error ? reason.message : 'Video connection failed',
        });
      }
    }
  }

  private isCurrent(generation: number): boolean {
    return this.generation === generation && this.references > 0;
  }

  private disconnect(): void {
    this.generation += 1;
    this.started = false;
    const peer = this.peer;
    this.peer = null;
    peer?.close();
    const sessionId = this.sessionId;
    this.sessionId = null;
    if (sessionId !== null) void this.releaseServerSession(sessionId, 'video session shutdown');
  }

  private async releaseServerSession(sessionId: string, context: string): Promise<void> {
    try {
      await this.transport.closeSession(sessionId);
    } catch (reason: unknown) {
      console.error(`Failed to release ${context}`, reason);
    }
  }
}

interface PoolEntry {
  readonly session: PooledVideoSession;
  closeTimer: number | null;
}

export class VideoSessionPool {
  private readonly entries = new Map<string, PoolEntry>();

  constructor(private readonly transport: VideoSessionTransport = defaultTransport) {}

  acquire(source: string, quality: VideoQuality): VideoSessionLease {
    const normalizedSource = source.trim();
    if (normalizedSource === '') throw new Error('Video source must not be empty');
    const key = `${quality}:${normalizedSource}`;
    let entry = this.entries.get(key);
    if (entry === undefined) {
      entry = { session: new PooledVideoSession(normalizedSource, quality, this.transport), closeTimer: null };
      this.entries.set(key, entry);
    }
    if (entry.closeTimer !== null) {
      window.clearTimeout(entry.closeTimer);
      entry.closeTimer = null;
    }
    const session = entry.session;
    session.retain();
    let released = false;
    return {
      getSnapshot: session.getSnapshot,
      subscribe: session.subscribe,
      retry: session.retry,
      release: () => {
        if (released) return;
        released = true;
        if (session.release() !== 0) return;
        entry.closeTimer = window.setTimeout(() => {
          entry!.closeTimer = null;
          if (session.referenceCount() !== 0) return;
          this.entries.delete(key);
          session.close();
        }, 0);
      },
    };
  }

  closeAll(): void {
    for (const entry of this.entries.values()) {
      if (entry.closeTimer !== null) window.clearTimeout(entry.closeTimer);
      entry.session.close();
    }
    this.entries.clear();
  }
}

export const videoSessionPool = new VideoSessionPool();
window.addEventListener('pagehide', (event) => {
  if (!event.persisted) videoSessionPool.closeAll();
});
