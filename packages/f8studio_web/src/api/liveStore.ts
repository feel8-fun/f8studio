import type { LiveSnapshot, LivePatch } from "./contracts.gen";
import { useCallback, useSyncExternalStore } from 'react';
import type { JsonValue } from './contracts';

type Listener = () => void;
interface Selection {
  snapshot: ReadonlyMap<string, JsonValue>;
  listeners: Set<Listener>;
}
function object(value: unknown): value is Record<string, JsonValue> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseLiveMessage(message: unknown): LiveSnapshot | LivePatch {
  if (!object(message)) throw new Error('Invalid live message');
  if (message.type === 'live.snapshot' && object(message.values)) {
    return { type: 'live.snapshot', values: message.values };
  }
  if (message.type === 'live.patch' && object(message.set) && Array.isArray(message.delete)) {
    const deleted: string[] = [];
    for (const key of message.delete) {
      if (typeof key !== 'string') throw new Error('Invalid live deletion');
      deleted.push(key);
    }
    return { type: 'live.patch', set: message.set, delete: deleted };
  }
  throw new Error('Invalid live message shape');
}

export class LiveStore {
  private values = new Map<string, JsonValue>();
  private selections = new Map<string, Selection>();
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private retry = 0;
  private users = 0;

  getPrefix(prefix: string): ReadonlyMap<string, JsonValue> {
    let selection = this.selections.get(prefix);
    if (selection === undefined) {
      selection = { snapshot: new Map([...this.values].filter(([key]) => key.startsWith(prefix))), listeners: new Set() };
      this.selections.set(prefix, selection);
    }
    return selection.snapshot;
  }

  subscribe(prefix: string, listener: Listener): () => void {
    this.getPrefix(prefix);
    this.selections.get(prefix)!.listeners.add(listener);
    this.users += 1;
    if (this.socket === null && this.timer === null) this.connect();
    return () => {
      const selection = this.selections.get(prefix);
      selection?.listeners.delete(listener);
      if (selection?.listeners.size === 0) this.selections.delete(prefix);
      this.users -= 1;
      if (this.users === 0) {
        if (this.timer !== null) clearTimeout(this.timer);
        this.timer = null;
        const socket = this.socket;
        this.socket = null;
        socket?.close();
        this.values.clear();
        this.selections.clear();
      }
    };
  }

  apply(raw: unknown): void {
    const message = parseLiveMessage(raw);
    const previous = this.values;
    let changed: Set<string>;
    if (message.type === 'live.snapshot') {
      this.values = new Map(Object.entries(message.values));
      changed = new Set([...previous.keys(), ...this.values.keys()]);
    } else {
      changed = new Set(Object.keys(message.set));
      for (const key of message.delete) {
        changed.add(key);
        this.values.delete(key);
      }
      for (const [key, value] of Object.entries(message.set)) this.values.set(key, value);
    }
    for (const [prefix, selection] of this.selections) {
      if (![...changed].some((key) => key.startsWith(prefix))) continue;
      selection.snapshot = new Map([...this.values].filter(([key]) => key.startsWith(prefix)));
      for (const listener of selection.listeners) listener();
    }
  }

  private connect(): void {
    if (this.users === 0) return;
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/live`);
    this.socket = socket;
    socket.onopen = () => { this.retry = 0; };
    socket.onmessage = (event) => {
      if (this.socket !== socket) return;
      try { this.apply(JSON.parse(String(event.data))); }
      catch (error) { console.error('Invalid Studio live data', error); socket.close(); }
    };
    socket.onerror = () => socket.close();
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.apply({ type: 'live.snapshot', values: {} });
      if (this.users > 0) this.timer = setTimeout(() => {
        this.timer = null;
        this.connect();
      }, Math.min(10_000, 250 * 2 ** this.retry++));
    };
  }
}

export const studioLive = new LiveStore();

export function useLivePrefix(prefix: string): ReadonlyMap<string, JsonValue> {
  const subscribe = useCallback((listener: Listener) => studioLive.subscribe(prefix, listener), [prefix]);
  const snapshot = useCallback(() => studioLive.getPrefix(prefix), [prefix]);
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}
