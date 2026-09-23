import { createContext, type ReactNode, useCallback, useContext, useEffect, useState, useSyncExternalStore } from 'react';

import { fetchPresentationSnapshot } from '../api/client';
import type { JsonValue, PresentationCommand } from '../api/contracts';
import { extensionRendererForCommand, extensionRendererById } from '../extensions/registry';

export type PresentationRenderer = 'text' | 'wave' | 'track' | 'video' | 'three_d' | (string & {});

export interface PresentationOutput {
  readonly nodeId: string;
  readonly renderer: PresentationRenderer;
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly updatedAt: number;
}

type Listener = () => void;

export function parsePresentationCommand(value: unknown): PresentationCommand | null {
  if (typeof value !== 'object' || value === null) return null;
  const envelope = value as Record<string, unknown>;
  if (envelope.type !== 'presentation.command' || typeof envelope.payload !== 'object' || envelope.payload === null) return null;
  const command = envelope.payload as Record<string, unknown>;
  if (typeof command.nodeId !== 'string' || typeof command.command !== 'string' || typeof command.payload !== 'object' || command.payload === null || Array.isArray(command.payload)) return null;
  return {
    nodeId: command.nodeId,
    command: command.command,
    payload: command.payload as Readonly<Record<string, JsonValue>>,
    tsMs: typeof command.tsMs === 'number' ? command.tsMs : null,
  };
}

function rendererFor(command: string): PresentationRenderer | null {
  if (command.startsWith('viz.text.')) return 'text';
  if (command.startsWith('viz.wave.')) return 'wave';
  if (command.startsWith('viz.track.')) return 'track';
  if (command.startsWith('viz.video.')) return 'video';
  if (command.startsWith('viz.three_d.')) return 'three_d';
  return extensionRendererForCommand(command)?.id ?? null;
}

export class PresentationStore {
  private readonly outputs = new Map<string, PresentationOutput>();
  private outputsSnapshot: ReadonlyMap<string, PresentationOutput> = new Map();
  private readonly outputListeners = new Set<Listener>();
  private readonly nodeListeners = new Map<string, Set<Listener>>();
  private readonly connectionListeners = new Set<Listener>();
  private socket: WebSocket | null = null;
  private retryTimer: number | null = null;
  private retryCount = 0;
  private started = false;
  private connected = false;
  private snapshotController: AbortController | null = null;

  readonly getOutputsSnapshot = (): ReadonlyMap<string, PresentationOutput> => this.outputsSnapshot;
  readonly getConnectionSnapshot = (): boolean => this.connected;

  getOutputSnapshot(nodeId: string): PresentationOutput | null {
    return this.outputs.get(nodeId) ?? null;
  }

  subscribeOutputs = (listener: Listener): (() => void) => {
    this.outputListeners.add(listener);
    return () => this.outputListeners.delete(listener);
  };

  subscribeConnection = (listener: Listener): (() => void) => {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  };

  subscribeNode(nodeId: string, listener: Listener): () => void {
    const existing = this.nodeListeners.get(nodeId);
    if (existing === undefined) this.nodeListeners.set(nodeId, new Set([listener]));
    else existing.add(listener);
    return () => {
      const listeners = this.nodeListeners.get(nodeId);
      listeners?.delete(listener);
      if (listeners?.size === 0) this.nodeListeners.delete(nodeId);
    };
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.connect();
  }

  stop(): void {
    this.started = false;
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.snapshotController?.abort();
    this.snapshotController = null;
    const socket = this.socket;
    this.socket = null;
    socket?.close();
    this.setConnected(false);
  }

  applyCommand(command: PresentationCommand): void {
    const renderer = rendererFor(command.command);
    if (renderer === null) return;
    if (command.command.endsWith('.detach')) {
      if (!this.outputs.delete(command.nodeId)) return;
      this.publishChanges([command.nodeId]);
      return;
    }

    const prior = this.outputs.get(command.nodeId);
    const updatedAt = command.tsMs ?? Date.now();
    if (prior !== undefined && prior.updatedAt > updatedAt) return;
    const priorPayload = prior?.renderer === renderer ? prior.payload : {};
    const extensionReducer = extensionRendererById(renderer)?.reduce;
    const payload = extensionReducer !== undefined
      ? extensionReducer(command.command, priorPayload, command.payload)
      : renderer === 'three_d' && command.command === 'viz.three_d.world_up'
        ? { ...priorPayload, ...command.payload }
        : command.payload;
    this.outputs.delete(command.nodeId);
    this.outputs.set(command.nodeId, { nodeId: command.nodeId, renderer, payload, updatedAt });

    const changedNodeIds = [command.nodeId];
    while (this.outputs.size > 32) {
      const oldestNodeId = this.outputs.keys().next().value;
      if (typeof oldestNodeId !== 'string') break;
      this.outputs.delete(oldestNodeId);
      changedNodeIds.push(oldestNodeId);
    }
    this.publishChanges(changedNodeIds);
  }

  private publishChanges(nodeIds: readonly string[]): void {
    this.outputsSnapshot = new Map(this.outputs);
    for (const listener of this.outputListeners) listener();
    for (const nodeId of new Set(nodeIds)) {
      for (const listener of this.nodeListeners.get(nodeId) ?? []) listener();
    }
  }

  private setConnected(connected: boolean): void {
    if (this.connected === connected) return;
    this.connected = connected;
    for (const listener of this.connectionListeners) listener();
  }

  private connect(): void {
    if (!this.started) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/events`);
    } catch (error: unknown) {
      console.error('Failed to open presentation event stream', error);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket !== socket) return;
      this.retryCount = 0;
      this.setConnected(true);
      this.snapshotController?.abort();
      const controller = new AbortController();
      this.snapshotController = controller;
      void fetchPresentationSnapshot(controller.signal).then(
        (commands) => {
          if (this.socket !== socket || controller.signal.aborted) return;
          for (const command of commands) this.applyCommand(command);
        },
        (reason: unknown) => {
          if (!controller.signal.aborted) console.error('Failed to load presentation snapshot', reason);
        },
      );
    };
    socket.onmessage = (event) => {
      if (this.socket !== socket) return;
      let decoded: unknown;
      try {
        decoded = JSON.parse(String(event.data));
      } catch (error: unknown) {
        console.error('Invalid presentation event JSON', error);
        return;
      }
      const command = parsePresentationCommand(decoded);
      if (command !== null) this.applyCommand(command);
    };
    socket.onerror = () => socket.close();
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.snapshotController?.abort();
      this.snapshotController = null;
      this.setConnected(false);
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (!this.started || this.retryTimer !== null) return;
    const delay = Math.min(5_000, 300 * 2 ** this.retryCount);
    this.retryCount += 1;
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }
}

const PresentationStoreContext = createContext<PresentationStore | null>(null);

export function PresentationProvider({ children }: { readonly children: ReactNode }) {
  const [store] = useState(() => new PresentationStore());
  useEffect(() => {
    store.start();
    return () => store.stop();
  }, [store]);
  return <PresentationStoreContext.Provider value={store}>{children}</PresentationStoreContext.Provider>;
}

function usePresentationStore(): PresentationStore {
  const store = useContext(PresentationStoreContext);
  if (store === null) throw new Error('Presentation hooks require PresentationProvider');
  return store;
}

export function usePresentationOutput(nodeId: string): PresentationOutput | null {
  const store = usePresentationStore();
  const subscribe = useCallback((listener: Listener) => store.subscribeNode(nodeId, listener), [nodeId, store]);
  const getSnapshot = useCallback(() => store.getOutputSnapshot(nodeId), [nodeId, store]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function usePresentationOutputs(): ReadonlyMap<string, PresentationOutput> {
  const store = usePresentationStore();
  return useSyncExternalStore(store.subscribeOutputs, store.getOutputsSnapshot, store.getOutputsSnapshot);
}

export function usePresentationConnected(): boolean {
  const store = usePresentationStore();
  return useSyncExternalStore(store.subscribeConnection, store.getConnectionSnapshot, store.getConnectionSnapshot);
}
