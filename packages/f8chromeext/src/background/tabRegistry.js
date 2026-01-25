import { t } from '../shared/i18n.js';

export class TabRegistry {
  constructor({ onChange } = {}) {
    this.tabs = new Map();
    this.preferredTabId = null;
    this.onChange = typeof onChange === 'function' ? onChange : () => {};
  }

  register(tabId, metadata = {}) {
    if (!tabId) return;
    const entry = {
      tabId,
      title: metadata.title || metadata.url || t('tab_registry_default_title'),
      url: metadata.url || '',
      registeredAt: Date.now(),
    };
    this.tabs.set(tabId, entry);
    if (!this.preferredTabId) {
      this.preferredTabId = tabId;
    }
    this.onChange(this.list());
    console.info('[Feel Bridge] Registered Feel tab', entry);
  }

  unregister(tabId) {
    if (!this.tabs.has(tabId)) {
      return null;
    }
    const entry = this.tabs.get(tabId);
    this.tabs.delete(tabId);
    if (this.preferredTabId === tabId) {
      this.preferredTabId = null;
    }
    this.ensurePreferred();
    this.onChange(this.list());
    console.info('[Feel Bridge] Unregistered Feel tab', entry);
    return entry;
  }

  ensurePreferred() {
    if (this.preferredTabId && this.tabs.has(this.preferredTabId)) {
      return this.preferredTabId;
    }
    const first = this.tabs.keys().next();
    this.preferredTabId = first.done ? null : first.value;
    return this.preferredTabId;
  }

  setPreferred(tabId) {
    if (tabId && this.tabs.has(tabId)) {
      this.preferredTabId = tabId;
    } else {
      this.preferredTabId = null;
    }
    this.ensurePreferred();
    this.onChange(this.list());
  }

  getPreferredTab() {
    const id = this.ensurePreferred();
    return id ? this.tabs.get(id) ?? null : null;
  }

  list() {
    return Array.from(this.tabs.values());
  }

  count() {
    return this.tabs.size;
  }

  has(tabId) {
    return this.tabs.has(tabId);
  }

  clear() {
    this.tabs.clear();
    this.preferredTabId = null;
    this.onChange([]);
  }
}
