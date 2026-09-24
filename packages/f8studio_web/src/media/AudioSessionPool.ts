import { closeAudioSession, createAudioSession } from '../api/client';
import { createRtcPeerConnection, waitForIceGatheringComplete } from './rtc';

export type AudioSessionSnapshot =
  | { readonly kind: 'connecting'; readonly stream: null }
  | { readonly kind: 'playing'; readonly stream: MediaStream }
  | { readonly kind: 'error'; readonly stream: null; readonly message: string };

type Listener = () => void;

export interface AudioSessionLease {
  readonly getSnapshot: () => AudioSessionSnapshot;
  readonly subscribe: (listener: Listener) => () => void;
  readonly retry: () => void;
  readonly release: () => void;
}

export interface AudioSessionTransport {
  readonly createPeer: () => Promise<RTCPeerConnection>;
  readonly waitForIce: (peer: RTCPeerConnection) => Promise<void>;
  readonly createSession: (source: string, description: RTCSessionDescriptionInit) => ReturnType<typeof createAudioSession>;
  readonly closeSession: (sessionId: string) => Promise<void>;
}

const defaultTransport: AudioSessionTransport = {
  createPeer: createRtcPeerConnection,
  waitForIce: waitForIceGatheringComplete,
  createSession: createAudioSession,
  closeSession: closeAudioSession,
};

const RELEASE_GRACE_MS = 1500;

class PooledAudioSession {
  private readonly listeners = new Set<Listener>();
  private snapshot: AudioSessionSnapshot = { kind: 'connecting', stream: null };
  private peer: RTCPeerConnection | null = null;
  private sessionId: string | null = null;
  private generation = 0;
  private references = 0;
  private started = false;

  constructor(readonly source: string, private readonly transport: AudioSessionTransport) {}

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

  referenceCount(): number { return this.references; }
  getSnapshot = (): AudioSessionSnapshot => this.snapshot;
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

  private setSnapshot(snapshot: AudioSessionSnapshot): void {
    this.snapshot = snapshot;
    for (const listener of this.listeners) listener();
  }

  private isCurrent(generation: number): boolean {
    return this.generation === generation && this.references > 0;
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
      activePeer.addTransceiver('audio', { direction: 'recvonly' });
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
      const answer = await this.transport.createSession(this.source, activePeer.localDescription);
      createdSessionId = answer.sessionId;
      if (!this.isCurrent(generation)) {
        activePeer.close();
        await this.releaseServerSession(createdSessionId);
        return;
      }
      this.sessionId = createdSessionId;
      await activePeer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    } catch (reason: unknown) {
      peer?.close();
      if (createdSessionId !== null && this.sessionId === createdSessionId) {
        this.sessionId = null;
        await this.releaseServerSession(createdSessionId);
      }
      if (this.isCurrent(generation)) {
        this.peer = null;
        this.setSnapshot({
          kind: 'error',
          stream: null,
          message: reason instanceof Error ? reason.message : 'Audio connection failed',
        });
      }
    }
  }

  private disconnect(): void {
    this.generation += 1;
    this.started = false;
    this.peer?.close();
    this.peer = null;
    const sessionId = this.sessionId;
    this.sessionId = null;
    if (sessionId !== null) void this.releaseServerSession(sessionId);
  }

  private async releaseServerSession(sessionId: string): Promise<void> {
    try {
      await this.transport.closeSession(sessionId);
    } catch (reason: unknown) {
      console.error('Failed to release audio session', reason);
    }
  }
}

interface PoolEntry {
  readonly session: PooledAudioSession;
  closeTimer: number | null;
}

export class AudioSessionPool {
  private readonly entries = new Map<string, PoolEntry>();

  constructor(private readonly transport: AudioSessionTransport = defaultTransport) {}

  acquire(source: string): AudioSessionLease {
    const normalizedSource = source.trim();
    if (normalizedSource === '') throw new Error('Audio source must not be empty');
    let entry = this.entries.get(normalizedSource);
    if (entry === undefined) {
      entry = { session: new PooledAudioSession(normalizedSource, this.transport), closeTimer: null };
      this.entries.set(normalizedSource, entry);
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
          this.entries.delete(normalizedSource);
          session.close();
        }, RELEASE_GRACE_MS);
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

export const audioSessionPool = new AudioSessionPool();
window.addEventListener('pagehide', (event) => {
  if (!event.persisted) audioSessionPool.closeAll();
});
