export class GatewaySessionStore {
  constructor() {
    this.sessionBySource = new Map();
    this.sourceBySession = new Map();
  }

  _newSessionId() {
    if (globalThis.crypto?.randomUUID) {
      return crypto.randomUUID();
    }
    return `s-${Math.random().toString(16).slice(2)}-${Date.now().toString(16)}`;
  }

  startForSource(sourceTabId) {
    if (!sourceTabId) return null;
    const existing = this.sessionBySource.get(sourceTabId);
    if (existing) return existing;
    const sessionId = this._newSessionId();
    this.sessionBySource.set(sourceTabId, sessionId);
    this.sourceBySession.set(sessionId, sourceTabId);
    return sessionId;
  }

  getSessionForSource(sourceTabId) {
    return this.sessionBySource.get(sourceTabId) ?? null;
  }

  getSourceForSession(sessionId) {
    return this.sourceBySession.get(sessionId) ?? null;
  }

  stopBySource(sourceTabId) {
    const sessionId = this.sessionBySource.get(sourceTabId);
    if (!sessionId) return null;
    this.sessionBySource.delete(sourceTabId);
    this.sourceBySession.delete(sessionId);
    return sessionId;
  }

  stopBySession(sessionId) {
    const sourceTabId = this.sourceBySession.get(sessionId);
    if (!sourceTabId) return null;
    this.sourceBySession.delete(sessionId);
    this.sessionBySource.delete(sourceTabId);
    return sourceTabId;
  }

  clearForTab(tabId) {
    this.stopBySource(tabId);
  }
}

