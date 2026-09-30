import { isJsonObject } from '../api/contracts';
import { createContext, type ReactNode, useCallback, useContext, useEffect, useState, useSyncExternalStore } from 'react';

import { fetchPresentationSnapshot } from '../api/client';
import { studioEvents } from '../api/eventStream';
import { studioLive } from '../api/liveStore';
import type { JsonValue, PresentationCommand } from '../api/contracts';
import { presentationRendererForCommand, presentationRendererById } from '../extensions/registry';

export type PresentationRenderer = 'text' | 'wave' | 'track' | 'video' | 'audio' | 'three_d' | (string & {});

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
  return presentationRendererForCommand(command)?.id ?? null;
}

export class PresentationStore {
  private readonly outputs = new Map<string, PresentationOutput>();
  private outputsSnapshot: ReadonlyMap<string, PresentationOutput> = new Map();
  private readonly outputListeners = new Set<Listener>();
  private readonly nodeListeners = new Map<string, Set<Listener>>();
  private readonly connectionListeners = new Set<Listener>();
  private subscriptions: (() => void)[] = [];
  private liveNodes = new Map<string, readonly JsonValue[]>();
  private eventConnected = false;
  private liveConnected = false;
  private arrivalSequence = 0;

  readonly getOutputsSnapshot = (): ReadonlyMap<string, PresentationOutput> => this.outputsSnapshot;
  readonly getConnectionSnapshot = (): boolean => this.eventConnected && this.liveConnected;

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
    if (this.subscriptions.length > 0) return;
    const updateLive = () => {
      const groups = new Map<string, JsonValue[]>();
      for (const value of studioLive.getPrefix('presentation/').values()) {
        if (!isJsonObject(value) || typeof value.nodeId !== 'string') continue;
        const group = groups.get(value.nodeId) ?? [];
        group.push(value);
        groups.set(value.nodeId, group);
      }
      for (const nodeId of this.liveNodes.keys()) {
        if (!groups.has(nodeId) && this.outputs.delete(nodeId)) this.publishChanges([nodeId]);
      }
      for (const [nodeId, values] of groups) {
        const previous = this.liveNodes.get(nodeId);
        if (previous?.length === values.length && previous.every((item, i) => item === values[i])) continue;
        this.outputs.delete(nodeId);
        values.sort((a, b) => {
          const first = a as Record<string, JsonValue>;
          const second = b as Record<string, JsonValue>;
          return Number(first.seq) - Number(second.seq);
        });
        for (const value of values) {
          const command = parsePresentationCommand({ type: 'presentation.command', payload: value });
          if (command !== null) this.applyCommand(command);
        }
      }
      this.liveNodes = groups;
    };
    this.subscriptions = [
      studioLive.subscribe('presentation/', updateLive),
      studioLive.subscribeConnection(() => this.setConnected('live', studioLive.getConnectionSnapshot())),
      studioEvents.subscribe((event) => {
        const command = parsePresentationCommand(event);
        if (command !== null) this.applyCommand(command);
      }, () => {
        // Extension commands without latest-value semantics retain their REST snapshot.
        void fetchPresentationSnapshot().then((commands) => {
          if (this.subscriptions.length === 0) return;
          for (const command of commands) {
            if (!presentationRendererForCommand(command.command)?.latestValue) this.applyCommand(command);
          }
        }).catch((error: unknown) => console.error('Failed to restore extension presentation', error));
      }, (connected) => this.setConnected('events', connected)),
    ];
    this.setConnected('live', studioLive.getConnectionSnapshot());
    updateLive();
  }

  stop(): void {
    for (const unsubscribe of this.subscriptions) unsubscribe();
    this.subscriptions = [];
    this.liveNodes.clear();
    this.setConnected('events', false);
    this.setConnected('live', false);
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
    const updatedAt = ++this.arrivalSequence;
    const priorPayload = prior?.renderer === renderer ? prior.payload : {};
    const reducer = presentationRendererById(renderer)?.reduce;
    const payload = reducer !== undefined
      ? reducer(command.command, priorPayload, command.payload)
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

  private setConnected(channel: 'events' | 'live', connected: boolean): void {
    const previous = this.getConnectionSnapshot();
    if (channel === 'events') this.eventConnected = connected;
    else this.liveConnected = connected;
    if (previous !== this.getConnectionSnapshot()) {
      for (const listener of this.connectionListeners) listener();
    }
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
