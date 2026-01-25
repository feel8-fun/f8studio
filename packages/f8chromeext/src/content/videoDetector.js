(async () => {
  const [{ BRIDGE_CONSTANTS, FEEL_SITE_URL }, { MESSAGE_TYPES }, runtime, i18n] = await Promise.all([
    import(chrome.runtime.getURL('src/shared/constants.js')),
    import(chrome.runtime.getURL('src/shared/messages.js')),
    import(chrome.runtime.getURL('src/shared/runtime.js')),
    import(chrome.runtime.getURL('src/shared/i18n.js')),
  ]);

  const { sendRuntimeMessage } = runtime;
  const { t } = i18n;
  const { FEEL_META_NAME, ICE_CONFIG, HIGHLIGHT_CLASS } = BRIDGE_CONSTANTS;

  const isFeelPlayer = () => !!document.querySelector(`meta[name="${FEEL_META_NAME}"]`);

  class VideoCatalog {
    constructor() {
      this.trackedVideos = new Map();
      this.videosById = new Map();
      this.counter = 0;
    }

    assignId(video) {
      if (!video.dataset.feelBridgeId) {
        video.dataset.feelBridgeId = `feel-video-${++this.counter}`;
      }
      return video.dataset.feelBridgeId;
    }

    track(video) {
      const id = this.assignId(video);
      this.videosById.set(id, video);
      if (!this.trackedVideos.has(video)) {
        this.trackedVideos.set(video, { id });
      }
    }

    untrack(video) {
      const info = this.trackedVideos.get(video);
      if (info?.id) {
        this.videosById.delete(info.id);
      }
      this.trackedVideos.delete(video);
      video.classList.remove(HIGHLIGHT_CLASS);
      delete video.dataset.feelBridgeId;
    }

    scan() {
      const videos = Array.from(document.querySelectorAll('video'));
      videos.forEach((video) => {
        if (!document.contains(video)) {
          this.untrack(video);
          return;
        }
        this.track(video);
      });

      Array.from(this.trackedVideos.keys()).forEach((video) => {
        if (!document.contains(video)) {
          this.untrack(video);
        }
      });
    }

    listMetadata() {
      this.scan();
      return Array.from(this.videosById.entries()).map(([id, video]) => this.toMetadata(video, id));
    }

    toMetadata(video, id = null) {
      return {
        id: id ?? video.dataset.feelBridgeId,
        src: video.currentSrc || video.src || (video.srcObject instanceof MediaStream ? t('popup_videos_live') : null),
        width: video.videoWidth,
        height: video.videoHeight,
        duration: Number.isFinite(video.duration) ? video.duration : null,
        muted: video.muted,
        paused: video.paused,
        readyState: video.readyState,
        poster: video.poster || null,
        playbackRate: video.playbackRate,
      };
    }

    getVideo(id) {
      return this.videosById.get(id) ?? null;
    }
  }

  class HighlightController {
    constructor(catalog) {
      this.catalog = catalog;
    }

    clear() {
      this.catalog.videosById.forEach((video) => video.classList.remove(HIGHLIGHT_CLASS));
    }

    setHighlight(id, active) {
      if (!id) {
        this.clear();
        return { ok: true };
      }
      const video = this.catalog.getVideo(id);
      if (!video) {
        return { ok: false, error: t('errors_video_not_found') };
      }
      if (active) {
        video.classList.add(HIGHLIGHT_CLASS);
        video.scrollIntoView({ block: 'center', behavior: 'smooth' });
      } else {
        video.classList.remove(HIGHLIGHT_CLASS);
      }
      return { ok: true };
    }
  }

  const captureVideoStream = (video) => {
    if (typeof video.captureStream === 'function') {
      return video.captureStream();
    }
    if (typeof video.mozCaptureStream === 'function') {
      return video.mozCaptureStream();
    }
    return null;
  };

  class SenderPeer {
    constructor(video, metadata, targetInfo) {
      this.video = video;
      this.metadata = metadata;
      this.target = targetInfo;
      this.stream = null;
      this.pc = new RTCPeerConnection(ICE_CONFIG);
      this.pc.onicecandidate = (event) => {
        if (event.candidate) {
          const candidateJson =
            typeof event.candidate.toJSON === 'function'
              ? event.candidate.toJSON()
              : {
                  candidate: event.candidate.candidate,
                  sdpMid: event.candidate.sdpMid,
                  sdpMLineIndex: event.candidate.sdpMLineIndex,
                  usernameFragment: event.candidate.usernameFragment,
                };
          chrome.runtime.sendMessage(
            {
              type: MESSAGE_TYPES.WEBRTC_ICE,
              payload: {
                candidate: candidateJson,
                targetKind: this.target.targetKind ?? null,
                targetTabId: this.target.targetTabId ?? null,
                sessionId: this.target.sessionId ?? null,
              },
            },
            () => void chrome.runtime.lastError,
          );
        }
      };
    }

    async start() {
      this.stream = captureVideoStream(this.video);
      if (!this.stream) {
        throw new Error(t('errors_video_uncapturable'));
      }
      this.stream.getTracks().forEach((track) => this.pc.addTrack(track, this.stream));
      try {
        await this.video.play();
      } catch {
        // captureStream still works even if autoplay is blocked
      }
      const offer = await this.pc.createOffer({
        offerToReceiveVideo: false,
        offerToReceiveAudio: false,
      });
      await this.pc.setLocalDescription(offer);
      const descriptionJson =
        typeof this.pc.localDescription?.toJSON === 'function'
          ? this.pc.localDescription.toJSON()
          : { type: this.pc.localDescription?.type, sdp: this.pc.localDescription?.sdp };
      await sendRuntimeMessage({
        type: MESSAGE_TYPES.WEBRTC_OFFER,
        payload: {
          targetKind: this.target.targetKind ?? null,
          targetTabId: this.target.targetTabId ?? null,
          sessionId: this.target.sessionId ?? null,
          description: descriptionJson,
          video: this.metadata,
          source: {
            title: document.title,
            url: window.location.href,
          },
        },
      });
    }

    async handleAnswer(description) {
      if (!description) return;
      await this.pc.setRemoteDescription(new RTCSessionDescription(description));
    }

    async handleRemoteIce(candidate) {
      if (!candidate) return;
      try {
        await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
      } catch (error) {
        console.warn('[Feel Bridge] Failed to add remote ICE candidate', error);
      }
    }

    stop(reason = 'stopped', notify = true) {
      this.stream?.getTracks().forEach((track) => track.stop());
      this.pc?.close();
      if (notify) {
        chrome.runtime.sendMessage(
          {
            type: MESSAGE_TYPES.WEBRTC_STOP,
            payload: {
              targetKind: this.target.targetKind ?? null,
              targetTabId: this.target.targetTabId ?? null,
              sessionId: this.target.sessionId ?? null,
              reason,
            },
          },
          () => void chrome.runtime.lastError,
        );
      }
    }
  }

  class SourceBridge {
    constructor(catalog) {
      this.catalog = catalog;
      this.highlightController = new HighlightController(catalog);
      this.currentSender = null;
    }

    init() {
      if (isFeelPlayer()) return;
    }

    dispose() {
      this.stopStreaming();
    }

    async collectMedia(sendResponse) {
      if (isFeelPlayer()) {
        sendResponse?.({ ok: false, error: t('errors_feel_tab_cannot_source') });
        return;
      }
      const videos = this.catalog.listMetadata();
      sendResponse?.({ ok: true, videos });
    }

    highlight(payload, sendResponse) {
      if (isFeelPlayer()) {
        sendResponse?.({ ok: false, error: t('errors_highlight_not_allowed') });
        return;
      }
      this.catalog.scan();
      const { id, active } = payload || {};
      const result = this.highlightController.setHighlight(id, active);
      sendResponse?.(result);
    }

    async getTargetInfo() {
      const response = await sendRuntimeMessage({ type: MESSAGE_TYPES.GET_TARGET });
      if (!response?.ok) {
        throw new Error(response?.error || t('errors_feel_tab_required', [FEEL_SITE_URL]));
      }
      if (response.targetKind === 'gateway') {
        return response;
      }
      if (!response.targetTabId) {
        throw new Error(t('errors_feel_tab_required', [FEEL_SITE_URL]));
      }
      return response;
    }

    async startStreaming(videoId, sendResponse) {
      if (isFeelPlayer()) {
        sendResponse?.({ ok: false, error: t('errors_feel_tab_cannot_source') });
        return;
      }
      this.catalog.scan();
      const video = this.catalog.getVideo(videoId);
      if (!video) {
        sendResponse?.({ ok: false, error: t('errors_video_not_found') });
        return;
      }
      try {
        const targetInfo = await this.getTargetInfo();
        this.stopStreaming('replaced');
        this.currentSender = new SenderPeer(video, this.catalog.toMetadata(video), targetInfo);
        await this.currentSender.start();
        sendResponse?.({ ok: true, targetTitle: targetInfo.targetTitle });
      } catch (error) {
        console.error('[Feel Bridge] Failed to start streaming', error);
        this.stopStreaming('error');
        sendResponse?.({ ok: false, error: error.message });
      }
    }

    stopStreaming(reason = 'stopped', notify = true, sendResponse) {
      if (this.currentSender) {
        this.currentSender.stop(reason, notify);
        this.currentSender = null;
      }
      sendResponse?.({ ok: true });
    }

    handleAnswer(payload) {
      this.currentSender?.handleAnswer(payload?.description);
    }

    handleIce(payload) {
      if (payload?.direction === 'receiver') {
        this.currentSender?.handleRemoteIce(payload?.candidate);
      }
    }
  }

  const bridge = new SourceBridge(new VideoCatalog());
  bridge.init();

  chrome.runtime?.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || typeof message !== 'object') return;
    switch (message.type) {
      case MESSAGE_TYPES.COLLECT_MEDIA:
        bridge.collectMedia(sendResponse);
        break;
      case MESSAGE_TYPES.HIGHLIGHT_VIDEO:
        bridge.highlight(message.payload, sendResponse);
        break;
      case MESSAGE_TYPES.START_STREAM:
        bridge.startStreaming(message.payload?.id, sendResponse);
        return true;
      case MESSAGE_TYPES.STOP_STREAM:
        bridge.stopStreaming('stopped', true, sendResponse);
        break;
      case MESSAGE_TYPES.WEBRTC_ANSWER:
        bridge.handleAnswer(message.payload);
        break;
      case MESSAGE_TYPES.WEBRTC_ICE:
        bridge.handleIce(message.payload);
        break;
      case MESSAGE_TYPES.WEBRTC_STOP:
        bridge.stopStreaming(message.payload?.reason || 'stopped', false, sendResponse);
        break;
      default:
        break;
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener(
      'DOMContentLoaded',
      () => {
        bridge.catalog.scan();
      },
      { once: true },
    );
  } else {
    bridge.catalog.scan();
  }
})();
