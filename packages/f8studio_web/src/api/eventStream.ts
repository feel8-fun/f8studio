import type { EventEnvelope } from './contracts.gen';

export type StudioEvent = EventEnvelope;
interface Consumer {
  readonly event: (event: StudioEvent) => void;
  readonly resync?: () => void;
  readonly connection?: (connected: boolean) => void;
}

function notifyConsumer(callback: (() => void) | undefined): void {
  try {
    callback?.();
  } catch (error) {
    console.error('Studio event consumer failed', error);
  }
}

/** One ordered event connection per browser window. */
export class EventStream {
  private consumers = new Set<Consumer>();
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private epoch: string | null = null;
  private sequence = 0;
  private retry = 0;
  private connected = false;

  subscribe(event: Consumer['event'], resync?: () => void, connection?: Consumer['connection']): () => void {
    const consumer: Consumer = { event, resync, connection };
    this.consumers.add(consumer);
    if (this.socket === null && this.timer === null) this.connect();
    // A new consumer has no local projection even if the shared stream resumed.
    queueMicrotask(() => {
      if (!this.consumers.has(consumer)) return;
      notifyConsumer(() => connection?.(this.connected));
      if (this.connected) notifyConsumer(resync);
    });
    return () => {
      this.consumers.delete(consumer);
      if (this.consumers.size === 0) {
        if (this.timer !== null) clearTimeout(this.timer);
        this.timer = null;
        const socket = this.socket;
        this.socket = null;
        this.connected = false;
        socket?.close();
      }
    };
  }

  private connect(): void {
    if (this.consumers.size === 0) return;
    const cursor = this.epoch === null ? '' : `?epoch=${encodeURIComponent(this.epoch)}&after=${this.sequence}`;
    let socket: WebSocket;
    try {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/events${cursor}`);
    } catch (error) {
      console.error('Studio event connection failed', error);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket !== socket) return;
      this.connected = true;
      this.retry = 0;
      for (const consumer of this.consumers) notifyConsumer(() => consumer.connection?.(true));
    };
    socket.onmessage = (message) => {
      if (this.socket !== socket) return;
      try {
        const value: unknown = JSON.parse(String(message.data));
        if (typeof value !== 'object' || value === null) throw new Error('Invalid event envelope');
        const item = value as Record<string, unknown>;
        if (typeof item.serverEpoch !== 'string' || typeof item.sequence !== 'number' || !Number.isSafeInteger(item.sequence)) throw new Error('Invalid event cursor');
        if (item.type === 'stream.hello') {
          if (item.resumed !== true) {
            this.epoch = item.serverEpoch;
            this.sequence = item.sequence;
            for (const consumer of this.consumers) notifyConsumer(consumer.resync);
          }
          // A resumed hello reports the server head, not the replay cursor.
          return;
        }
        if (item.serverEpoch !== this.epoch || item.sequence <= this.sequence) return;
        if (typeof item.type !== 'string' || typeof item.scope !== 'string') throw new Error('Invalid event type/scope');
        for (const consumer of this.consumers) notifyConsumer(() => consumer.event(item as unknown as StudioEvent));
        this.sequence = item.sequence;
      } catch (error) {
        console.error('Studio event delivery failed; reconnecting from the last processed event', error);
        socket.close();
      }
    };
    socket.onerror = () => socket.close();
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.connected = false;
      for (const consumer of this.consumers) notifyConsumer(() => consumer.connection?.(false));
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.consumers.size > 0 && this.timer === null) this.timer = setTimeout(() => {
        this.timer = null;
        this.connect();
      }, Math.min(10_000, 250 * 2 ** this.retry++));
  }
}

export const studioEvents = new EventStream();
