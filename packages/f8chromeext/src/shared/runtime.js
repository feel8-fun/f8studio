export const safeNotifyTab = (tabId, message) => {
  if (!tabId) return;
  chrome.tabs.sendMessage(tabId, message, () => void chrome.runtime.lastError);
};

export const safeSendMessage = (message) => {
  try {
    chrome.runtime?.sendMessage?.(message);
  } catch (error) {
    if (error?.message?.includes('Extension context invalidated')) {
      return;
    }
    console.warn('[Feel Bridge] Runtime message failed', error);
  }
};

export const sendRuntimeMessage = (message) =>
  new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      resolve(response);
    });
  });

export const sendTabMessage = (tabId, message, options = {}) =>
  new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, message, options, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response);
    });
  });
