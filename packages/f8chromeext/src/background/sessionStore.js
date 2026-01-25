import { safeNotifyTab } from '../shared/runtime.js';
import { MESSAGE_TYPES } from '../shared/messages.js';

export class SessionStore {
  constructor() {
    this.sessionsBySource = new Map();
    this.sessionsByTarget = new Map();
  }

  attach(sourceId, targetId) {
    if (!sourceId || !targetId) return;
    const existingSource = this.sessionsByTarget.get(targetId);
    if (existingSource && existingSource !== sourceId) {
      this.stopBySource(existingSource, 'replaced');
    }
    const existingTarget = this.sessionsBySource.get(sourceId);
    if (existingTarget && existingTarget !== targetId) {
      this.stopByTarget(existingTarget, 'replaced');
    }

    this.sessionsBySource.set(sourceId, targetId);
    this.sessionsByTarget.set(targetId, sourceId);
  }

  stopBySource(sourceId, reason = 'stopped') {
    if (!this.sessionsBySource.has(sourceId)) return;
    const targetId = this.sessionsBySource.get(sourceId);
    this.sessionsBySource.delete(sourceId);
    if (targetId) {
      this.sessionsByTarget.delete(targetId);
      safeNotifyTab(targetId, { type: MESSAGE_TYPES.WEBRTC_STOP, payload: { reason } });
    }
  }

  stopByTarget(targetId, reason = 'stopped') {
    if (!this.sessionsByTarget.has(targetId)) return;
    const sourceId = this.sessionsByTarget.get(targetId);
    this.sessionsByTarget.delete(targetId);
    if (sourceId) {
      this.sessionsBySource.delete(sourceId);
      safeNotifyTab(sourceId, { type: MESSAGE_TYPES.WEBRTC_STOP, payload: { reason } });
    }
  }

  getTargetForSource(sourceId) {
    return this.sessionsBySource.get(sourceId) ?? null;
  }

  getSourceForTarget(targetId) {
    return this.sessionsByTarget.get(targetId) ?? null;
  }

  clearForTab(tabId) {
    this.stopBySource(tabId, 'stopped');
    this.stopByTarget(tabId, 'stopped');
  }
}
