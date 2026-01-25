import { MESSAGE_TYPES } from '../shared/messages.js';
import { sendTabMessage } from '../shared/runtime.js';
import { t } from '../shared/i18n.js';

class PopupController {
  constructor() {
    this.statusEl = document.getElementById('status');
    this.listEl = document.getElementById('video-list');
    this.itemTemplate = document.getElementById('video-item-template');
    this.activeTabId = null;
    this.highlightTimer = null;
    this.applyStaticTranslations();
  }

  applyStaticTranslations() {
    const titleEl = document.querySelector('[data-i18n-key="popup_title"]');
    if (titleEl) {
      titleEl.textContent = t('popup_title');
    }
    const subtitleEl = document.querySelector('[data-i18n-key="popup_subtitle"]');
    if (subtitleEl) {
      subtitleEl.textContent = t('popup_subtitle');
    }
  }

  setStatus(text) {
    if (!this.statusEl) return;
    this.statusEl.textContent = text ?? '';
    this.statusEl.hidden = !text;
  }

  async resolveActiveTab() {
    return new Promise((resolve) => {
      chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
        this.activeTabId = tab?.id ?? null;
        resolve(this.activeTabId);
      });
    });
  }

  async highlightVideo(video, active) {
    if (!this.activeTabId) return;
    const payload = { id: video?.id ?? null, active };
    const options = video?.frameId !== undefined ? { frameId: video.frameId } : {};
    await sendTabMessage(
      this.activeTabId,
      {
        type: MESSAGE_TYPES.HIGHLIGHT_VIDEO,
        payload,
      },
      options,
    );
  }

  async clearHighlight() {
    await this.highlightVideo(null, false);
  }

  async startStream(video) {
    if (!this.activeTabId) {
      this.setStatus(t('popup_status_no_active_tab'));
      return;
    }
    this.setStatus(t('popup_status_connecting'));
    const options = video?.frameId !== undefined ? { frameId: video.frameId } : {};
    const response = await sendTabMessage(
      this.activeTabId,
      {
        type: MESSAGE_TYPES.START_STREAM,
        payload: { id: video.id },
      },
      options,
    );
    if (!response?.ok) {
      const reason = response?.error || t('popup_status_unknown_error');
      this.setStatus(t('popup_status_failed', [reason]));
      return;
    }
    const label = response.targetTitle || t('popup_target_fallback');
    this.setStatus(t('popup_status_streaming', [label]));
    if (this.highlightTimer) clearTimeout(this.highlightTimer);
    this.highlightTimer = setTimeout(() => this.setStatus(''), 2500);
  }

  formatVideoMeta(video) {
    const resolution =
      video.width && video.height
        ? `${video.width}x${video.height}`
        : t('popup_videos_unknown_size');
    const duration =
      typeof video.duration === 'number'
        ? t('popup_videos_duration_seconds', [String(Math.round(video.duration))])
        : video.readyState >= 2
          ? t('popup_videos_live')
          : t('popup_videos_unknown_duration');
    return `${resolution} · ${duration}`;
  }

  renderVideos(videos) {
    this.listEl.innerHTML = '';
    if (!videos.length) {
      this.setStatus(t('popup_status_no_videos'));
      return;
    }
    this.setStatus('');

    videos.forEach((video, index) => {
      const fragment = this.itemTemplate.content.cloneNode(true);
      const itemEl = fragment.querySelector('.video-item');
      fragment.querySelector('.video-index').textContent = t('popup_videos_index_label', [
        String(index + 1),
      ]);
      fragment.querySelector('.video-meta').textContent = this.formatVideoMeta(video);
      fragment.querySelector('.video-src').textContent =
        video.src || t('popup_videos_inline_source');
      const streamButton = fragment.querySelector('.stream-button');
      streamButton.textContent = t('popup_button_stream');
      streamButton.addEventListener('click', () => {
        this.startStream(video).catch((error) => {
          console.error('[Feel Bridge] Failed to start stream from popup', error);
          const reason = error?.message || String(error);
          this.setStatus(t('popup_status_failed', [reason]));
        });
      });
      itemEl.addEventListener('mouseenter', () => this.highlightVideo(video, true));
      itemEl.addEventListener('mouseleave', () => this.highlightVideo(video, false));
      this.listEl.appendChild(fragment);
    });
  }

  async getFramesForTab(tabId) {
    if (!chrome.webNavigation?.getAllFrames) {
      return [{ frameId: 0 }];
    }
    try {
      const frames = await chrome.webNavigation.getAllFrames({ tabId });
      if (!frames || !frames.length) {
        return [{ frameId: 0 }];
      }
      return frames.map((frame) => ({ frameId: frame.frameId }));
    } catch (error) {
      console.warn('[Feel Bridge] Failed to enumerate frames', error);
      return [{ frameId: 0 }];
    }
  }

  async collectVideos(tabId) {
    const frames = await this.getFramesForTab(tabId);
    const videos = [];
    for (const frame of frames) {
      const response = await sendTabMessage(
        tabId,
        { type: MESSAGE_TYPES.COLLECT_MEDIA },
        { frameId: frame.frameId },
      );
      if (response?.ok && Array.isArray(response.videos)) {
        response.videos.forEach((video) => {
          videos.push({ ...video, frameId: frame.frameId });
        });
      }
    }
    return videos;
  }

  async loadVideos() {
    const tabId = await this.resolveActiveTab();
    if (!tabId) {
      this.setStatus(t('popup_status_no_active_tab'));
      return;
    }
    const videos = await this.collectVideos(tabId);
    this.renderVideos(videos);
  }
}

const popup = new PopupController();
popup.setStatus(t('popup_status_detecting'));
popup.loadVideos().catch((error) => {
  const message = error?.message || String(error);
  popup.setStatus(t('popup_status_error', [message]));
});

window.addEventListener('unload', () => {
  popup.clearHighlight().catch(() => {});
});
