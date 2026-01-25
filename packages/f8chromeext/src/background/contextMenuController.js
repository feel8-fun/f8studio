import { t } from '../shared/i18n.js';

export class ContextMenuController {
  constructor(registry) {
    this.registry = registry;
  }

  rebuild() {
    if (!chrome.contextMenus?.removeAll) return;
    chrome.contextMenus.removeAll(() => {
      chrome.contextMenus.create({
        id: 'feel-top-sep',
        type: 'separator',
        contexts: ['action'],
      });

      const entries = this.registry.list();
      if (!entries.length) {
        chrome.contextMenus.create({
          id: 'feel-empty',
          title: t('context_menu_no_tab'),
          contexts: ['action'],
          enabled: false,
        });
        return;
      }

      const preferred = this.registry.getPreferredTab();
      chrome.contextMenus.create({
        id: 'feel-group',
        title: preferred
          ? t('context_menu_group_title_with_target', [preferred.title])
          : t('context_menu_group_title'),
        contexts: ['action'],
        enabled: true,
      });

      entries.forEach((tab) => {
        chrome.contextMenus.create({
          id: `select-${tab.tabId}`,
          parentId: 'feel-group',
          title: tab.title || tab.url || t('context_menu_tab_fallback', [String(tab.tabId)]),
          contexts: ['action'],
          type: 'radio',
          checked: preferred ? tab.tabId === preferred.tabId : false,
        });
      });
    });
  }

  handleClick(info) {
    if (!info.menuItemId) return;
    if (!info.menuItemId.startsWith('select-')) return;
    const tabId = Number(info.menuItemId.replace('select-', ''));
    this.registry.setPreferred(tabId);
    this.rebuild();
    if (this.registry.has(tabId)) {
      chrome.tabs.get(tabId, (tab) => {
        if (chrome.runtime.lastError) return;
        chrome.windows.update(tab.windowId, { focused: true }, () => void chrome.runtime.lastError);
        chrome.tabs.update(tabId, { active: true }, () => void chrome.runtime.lastError);
      });
    }
  }
}
