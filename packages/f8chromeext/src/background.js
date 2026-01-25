import { TabRegistry } from './background/tabRegistry.js';
import { SessionStore } from './background/sessionStore.js';
import { ContextMenuController } from './background/contextMenuController.js';
import { GatewaySessionStore } from './background/gatewaySessionStore.js';
import { GatewayWsClient } from './background/gatewayWsClient.js';
import { MESSAGE_TYPES } from './shared/messages.js';
import { safeNotifyTab } from './shared/runtime.js';
import { BRIDGE_CONSTANTS, FEEL_SITE_URL } from './shared/constants.js';
import { t } from './shared/i18n.js';

const registry = new TabRegistry({
  onChange: () => {
    updateBadge();
    menuController.rebuild();
  },
});
const sessions = new SessionStore();
const gatewaySessions = new GatewaySessionStore();
const menuController = new ContextMenuController(registry);
const DEFAULT_CLIENT_WAIT_MS = 20_000;
let pendingFeelTabPromise = null;
const pendingGatewayIce = new Map();

const gatewayClient = new GatewayWsClient(BRIDGE_CONSTANTS.GATEWAY_WS_URL, {
  onMessage: (text) => handleGatewayWsMessage(text),
  onOpen: () => {
    gatewayClient.sendJson({
      type: 'hello',
      client: 'f8chromeext',
      ts: Date.now(),
    });
  },
});

const waitForPreferredTab = (timeoutMs = DEFAULT_CLIENT_WAIT_MS) =>
  new Promise((resolve, reject) => {
    const startedAt = Date.now();
    let retryTimer = null;

    const cleanup = () => {
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
    };

    const check = () => {
      const target = registry.getPreferredTab();
      if (target) {
        cleanup();
        resolve(target);
        return;
      }
      if (Date.now() - startedAt >= timeoutMs) {
        cleanup();
        reject(new Error(t('errors_feel_tab_loading', [FEEL_SITE_URL])));
        return;
      }
      retryTimer = setTimeout(check, 250);
    };

    check();
  });

const openDefaultFeelTab = () =>
  new Promise((resolve, reject) => {
    chrome.tabs.create({ url: FEEL_SITE_URL, active: false }, (tab) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err?.message || t('errors_open_feel_tab_failed')));
        return;
      }
      if (!tab?.id) {
        reject(new Error(t('errors_open_feel_tab_failed')));
        return;
      }
      resolve(tab.id);
    });
  });

const ensurePreferredTabAvailable = async () => {
  const existing = registry.getPreferredTab();
  if (existing) {
    return existing;
  }
  if (!pendingFeelTabPromise) {
    pendingFeelTabPromise = (async () => {
    console.info('[Feel Bridge] No Feel tab detected. Opening default client tab:', FEEL_SITE_URL);
      await openDefaultFeelTab();
      return waitForPreferredTab();
    })()
      .catch((error) => {
        console.warn('[Feel Bridge] Failed to prepare Feel tab automatically', error);
        throw error;
      })
      .finally(() => {
        pendingFeelTabPromise = null;
      });
  }
  return pendingFeelTabPromise;
};

function updateBadge() {
  if (!chrome.action || !chrome.action.setBadgeText) return;
  const count = registry.count();
  chrome.action.setBadgeBackgroundColor?.({ color: '#6366f1' });
  chrome.action.setBadgeText({ text: count ? String(count) : '' });
}

const ensurePreferredTabId = () => registry.ensurePreferred();

const handleRegisterPlayer = (sender, payload, sendResponse) => {
  const tabId = sender.tab?.id;
  if (!tabId) {
    sendResponse?.({ ok: false, error: t('errors_missing_tab_context') });
    return;
  }
  registry.register(tabId, payload);
  sendResponse?.({ ok: true });
};

const handleUnregisterPlayer = (sender, sendResponse) => {
  const tabId = sender.tab?.id;
  if (!tabId) {
    sendResponse?.({ ok: false, error: t('errors_missing_tab_context') });
    return;
  }
  registry.unregister(tabId);
  sessions.stopByTarget(tabId, 'stopped');
  sendResponse?.({ ok: true });
};

const handleGetTarget = async (sendResponse) => {
  const prefer = BRIDGE_CONSTANTS.DEFAULT_SIGNALING_TARGET || 'feel';
  if (prefer !== 'feel' && BRIDGE_CONSTANTS.GATEWAY_WS_URL) {
    const ok = await gatewayClient.ensureOpen(800);
    if (ok) {
      sendResponse?.({
        ok: true,
        targetKind: 'gateway',
        targetTitle: t('popup_target_gateway'),
        wsUrl: BRIDGE_CONSTANTS.GATEWAY_WS_URL,
      });
      return;
    }
    if (prefer === 'gateway') {
      sendResponse?.({
        ok: false,
        error: t('errors_gateway_unavailable', [BRIDGE_CONSTANTS.GATEWAY_WS_URL]),
      });
      return;
    }
  }
  try {
    const preferred = await ensurePreferredTabAvailable();
    sendResponse?.({
      ok: true,
      targetKind: 'feelTab',
      targetTabId: preferred.tabId,
      targetTitle: preferred.title,
    });
  } catch (error) {
    sendResponse?.({
      ok: false,
      error: error?.message || t('errors_feel_tab_required', [FEEL_SITE_URL]),
    });
  }
};

const forwardOfferToTarget = (sourceTabId, payload, sendResponse) => {
  const targetTabId = payload?.targetTabId;
  if (!targetTabId || !sourceTabId) {
    sendResponse?.({ ok: false, error: t('errors_missing_target_tab') });
    return;
  }
  sessions.attach(sourceTabId, targetTabId);
  chrome.tabs.sendMessage(
    targetTabId,
    {
      type: MESSAGE_TYPES.WEBRTC_OFFER,
      payload: {
        description: payload?.description,
        video: payload?.video || null,
        source: {
          tabId: sourceTabId,
          title: payload?.source?.title,
          url: payload?.source?.url,
        },
      },
    },
    () => {
      const err = chrome.runtime.lastError;
      if (err && !err.message?.includes('The message port closed before a response was received.')) {
        console.warn('[Feel Bridge] Failed to deliver offer:', err.message);
        sessions.stopBySource(sourceTabId, 'error');
        sendResponse?.({ ok: false, error: err.message });
        return;
      }
      sendResponse?.({ ok: true });
    },
  );
};

const forwardOfferToGateway = async (sourceTabId, payload, sendResponse) => {
  if (!sourceTabId) {
    sendResponse?.({ ok: false, error: t('errors_missing_tab_context') });
    return;
  }

  const sessionId = gatewaySessions.startForSource(sourceTabId);
  const ok = await gatewayClient.ensureOpen(800);
  if (!ok) {
    sendResponse?.({
      ok: false,
      error: t('errors_gateway_unavailable', [BRIDGE_CONSTANTS.GATEWAY_WS_URL]),
    });
    return;
  }

  gatewayClient.sendJson({
    type: 'webrtc.offer',
    sessionId,
    description: payload?.description ?? null,
    video: payload?.video ?? null,
    source: {
      tabId: sourceTabId,
      title: payload?.source?.title ?? null,
      url: payload?.source?.url ?? null,
    },
    ts: Date.now(),
  });

  const pending = pendingGatewayIce.get(sourceTabId);
  if (pending?.length) {
    pendingGatewayIce.delete(sourceTabId);
    pending.slice(0, 64).forEach((candidate) => {
      gatewayClient.sendJson({
        type: 'webrtc.ice',
        sessionId,
        candidate,
        ts: Date.now(),
      });
    });
  }

  sendResponse?.({
    ok: true,
    targetKind: 'gateway',
    targetTitle: t('popup_target_gateway'),
    sessionId,
  });
};

const handleWebrtcAnswer = (sender, payload, sendResponse) => {
  const targetId = sender.tab?.id;
  const sourceId = targetId ? sessions.getSourceForTarget(targetId) : null;
  if (!sourceId) {
    sendResponse?.({ ok: false, error: t('errors_no_pending_session') });
    return;
  }
  safeNotifyTab(sourceId, {
    type: MESSAGE_TYPES.WEBRTC_ANSWER,
    payload: { description: payload?.description },
  });
  sendResponse?.({ ok: true });
};

const handleIceCandidate = (sender, payload, sendResponse) => {
  const senderId = sender.tab?.id;
  if (!senderId || !payload?.candidate) {
    sendResponse?.({ ok: false, error: t('errors_invalid_ice_payload') });
    return;
  }
  if (sessions.getTargetForSource(senderId)) {
    const targetId = sessions.getTargetForSource(senderId);
    safeNotifyTab(targetId, {
      type: MESSAGE_TYPES.WEBRTC_ICE,
      payload: { candidate: payload.candidate, direction: 'sender' },
    });
    sendResponse?.({ ok: true });
    return;
  }
  if (sessions.getSourceForTarget(senderId)) {
    const sourceId = sessions.getSourceForTarget(senderId);
    safeNotifyTab(sourceId, {
      type: MESSAGE_TYPES.WEBRTC_ICE,
      payload: { candidate: payload.candidate, direction: 'receiver' },
    });
    sendResponse?.({ ok: true });
    return;
  }

  const sessionId = gatewaySessions.getSessionForSource(senderId);
  if (sessionId) {
    gatewayClient.sendJson({
      type: 'webrtc.ice',
      sessionId,
      candidate: payload.candidate,
      ts: Date.now(),
    });
    sendResponse?.({ ok: true });
    return;
  }

  if (payload?.targetKind === 'gateway') {
    const existing = pendingGatewayIce.get(senderId) || [];
    existing.push(payload.candidate);
    pendingGatewayIce.set(senderId, existing.slice(-64));
    sendResponse?.({ ok: true });
    return;
  }

  sendResponse?.({ ok: false, error: t('errors_no_active_session') });
};

const handleStop = (sender, payload, sendResponse) => {
  const senderId = sender.tab?.id;
  if (!senderId) {
    sendResponse?.({ ok: false });
    return;
  }
  const reason = payload?.reason || 'stopped';
  if (sessions.getTargetForSource(senderId)) {
    sessions.stopBySource(senderId, reason);
  } else if (sessions.getSourceForTarget(senderId)) {
    sessions.stopByTarget(senderId, reason);
  } else if (gatewaySessions.getSessionForSource(senderId)) {
    const sessionId = gatewaySessions.stopBySource(senderId);
    pendingGatewayIce.delete(senderId);
    if (sessionId) {
      gatewayClient.sendJson({
        type: 'webrtc.stop',
        sessionId,
        reason,
        ts: Date.now(),
      });
    }
  }
  sendResponse?.({ ok: true });
};

function handleGatewayWsMessage(text) {
  if (!text) return;
  let msg = null;
  try {
    msg = JSON.parse(text);
  } catch {
    return;
  }
  if (!msg || typeof msg !== 'object') return;

  const type = msg.type || msg.event || null;
  const payload = msg.payload && typeof msg.payload === 'object' ? msg.payload : msg;
  const sessionId = payload.sessionId || msg.sessionId || null;
  if (!type || !sessionId) return;

  const sourceTabId = gatewaySessions.getSourceForSession(sessionId);
  if (!sourceTabId) return;

  if (type === 'webrtc.answer') {
    safeNotifyTab(sourceTabId, {
      type: MESSAGE_TYPES.WEBRTC_ANSWER,
      payload: { description: payload.description ?? null },
    });
    return;
  }

  if (type === 'webrtc.ice') {
    safeNotifyTab(sourceTabId, {
      type: MESSAGE_TYPES.WEBRTC_ICE,
      payload: { candidate: payload.candidate ?? null, direction: 'receiver' },
    });
    return;
  }

  if (type === 'webrtc.stop') {
    gatewaySessions.stopBySession(sessionId);
    safeNotifyTab(sourceTabId, {
      type: MESSAGE_TYPES.WEBRTC_STOP,
      payload: { reason: payload.reason || 'stopped' },
    });
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || typeof message !== 'object') return;
  const { type, payload } = message;

  switch (type) {
    case MESSAGE_TYPES.REGISTER_PLAYER:
      handleRegisterPlayer(sender, payload, sendResponse);
      break;
    case MESSAGE_TYPES.UNREGISTER_PLAYER:
      handleUnregisterPlayer(sender, sendResponse);
      break;
    case MESSAGE_TYPES.LIST_PLAYERS:
      sendResponse?.({ ok: true, tabs: registry.list() });
      break;
    case MESSAGE_TYPES.GET_TARGET:
      handleGetTarget(sendResponse);
      return true;
    case MESSAGE_TYPES.WEBRTC_OFFER: {
      const sourceId = sender.tab?.id;
      if (payload?.targetTabId) {
        forwardOfferToTarget(sourceId, payload, sendResponse);
        return true;
      }
      forwardOfferToGateway(sourceId, payload, sendResponse);
      return true;
    }
    case MESSAGE_TYPES.WEBRTC_ANSWER:
      handleWebrtcAnswer(sender, payload, sendResponse);
      break;
    case MESSAGE_TYPES.WEBRTC_ICE:
      handleIceCandidate(sender, payload, sendResponse);
      break;
    case MESSAGE_TYPES.WEBRTC_STOP:
      handleStop(sender, payload, sendResponse);
      break;
    default:
      break;
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  registry.unregister(tabId);
  sessions.clearForTab(tabId);
  gatewaySessions.clearForTab(tabId);
  pendingGatewayIce.delete(tabId);
});

chrome.runtime.onInstalled.addListener(() => {
  menuController.rebuild();
});

chrome.contextMenus?.onClicked.addListener((info) => {
  menuController.handleClick(info);
});

ensurePreferredTabId();
updateBadge();
menuController.rebuild();
gatewayClient.connect();
