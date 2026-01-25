export class GatewayWsClient {
  constructor(url, { onMessage, onOpen, onClose } = {}) {
    this.url = url;
    this.onMessage = typeof onMessage === 'function' ? onMessage : null;
    this.onOpen = typeof onOpen === 'function' ? onOpen : null;
    this.onClose = typeof onClose === 'function' ? onClose : null;

    this.ws = null;
    this.state = 'idle';
    this.reconnectTimer = null;
    this.backoffMs = 250;
    this.outbox = [];
  }

  isOpen() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  connect() {
    if (this.isOpen() || this.state === 'connecting') return;
    if (!this.url) return;

    this.state = 'connecting';
    try {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      ws.onopen = () => {
        this.state = 'open';
        this.backoffMs = 250;
        this._flushOutbox();
        this.onOpen?.();
      };
      ws.onclose = () => {
        const wasOpen = this.state === 'open';
        this.state = 'closed';
        this.ws = null;
        if (wasOpen) this.onClose?.();
        this._scheduleReconnect();
      };
      ws.onerror = () => {
        // onclose will handle reconnection.
      };
      ws.onmessage = (event) => {
        if (!event?.data) return;
        this.onMessage?.(String(event.data));
      };
    } catch {
      this.state = 'closed';
      this.ws = null;
      this._scheduleReconnect();
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    try {
      this.ws?.close();
    } catch {
      // ignore
    } finally {
      this.ws = null;
      this.state = 'closed';
    }
  }

  async ensureOpen(timeoutMs = 1000) {
    if (this.isOpen()) return true;
    this.connect();
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      if (this.isOpen()) return true;
      await new Promise((r) => setTimeout(r, 50));
    }
    return this.isOpen();
  }

  sendText(text) {
    if (!text) return;
    if (this.isOpen()) {
      try {
        this.ws.send(text);
        return;
      } catch {
        // fallthrough to queue
      }
    }
    this.outbox.push(String(text));
    this.connect();
  }

  sendJson(obj) {
    this.sendText(JSON.stringify(obj));
  }

  _flushOutbox() {
    if (!this.isOpen() || !this.outbox.length) return;
    const pending = this.outbox.splice(0, this.outbox.length);
    pending.forEach((text) => {
      try {
        this.ws.send(text);
      } catch {
        // drop on send failure
      }
    });
  }

  _scheduleReconnect() {
    if (this.reconnectTimer) return;
    const delay = this.backoffMs;
    this.backoffMs = Math.min(this.backoffMs * 2, 5000);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }
}

