(async () => {
  const [{ BRIDGE_CONSTANTS, FEEL_SITE_URL }, { MESSAGE_TYPES }, runtime, i18n] = await Promise.all([
    import(chrome.runtime.getURL('src/shared/constants.js')),
    import(chrome.runtime.getURL('src/shared/messages.js')),
    import(chrome.runtime.getURL('src/shared/runtime.js')),
    import(chrome.runtime.getURL('src/shared/i18n.js')),
  ]);

  const { sendRuntimeMessage } = runtime;
  const { t } = i18n;
  const { FEEL_META_NAME, ICE_CONFIG, HIGHLIGHT_CLASS, GATEWAY_VIDEO_CODEC } = BRIDGE_CONSTANTS;

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
      this.gatewayWs = null;
      this.keepalivePort = null;
      this.statsInterval = null;
      this.gatewaySessionId =
        this.target?.targetKind === 'gateway'
          ? (globalThis.crypto?.randomUUID?.() ||
              `gw_${Date.now()}_${Math.random().toString(16).slice(2)}_${Math.random().toString(16).slice(2)}`)
          : null;
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
          if (this.target?.targetKind === 'gateway' && this.gatewayWs?.readyState === WebSocket.OPEN) {
            try {
              this.gatewayWs.send(
                JSON.stringify({
                  type: 'webrtc.ice',
                  sessionId: this.gatewaySessionId,
                  candidate: candidateJson,
                  ts: Date.now(),
                }),
              );
            } catch {
              // ignore
            }
          } else {
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
        }
      };
    }

    _applyVideoCodecPreferences(preferCodec) {
      const prefer = typeof preferCodec === 'string' ? preferCodec.trim() : '';
      if (!prefer || prefer.toLowerCase() === 'auto') return;

      const transceiver = this.pc
        ?.getTransceivers?.()
        ?.find((t) => t?.sender?.track?.kind === 'video' && typeof t?.setCodecPreferences === 'function');
      if (!transceiver) return;

      const caps = globalThis.RTCRtpSender?.getCapabilities?.('video');
      const codecs = Array.isArray(caps?.codecs) ? caps.codecs : [];
      if (!codecs.length) return;

      const wantMime = `video/${prefer.toLowerCase()}`;
      const preferred = codecs.filter((c) => String(c?.mimeType || '').toLowerCase() === wantMime);
      if (!preferred.length) return;

      const rtx = codecs.filter((c) => String(c?.mimeType || '').toLowerCase() === 'video/rtx');
      transceiver.setCodecPreferences([...preferred, ...rtx]);
    }

    _startStatsLog() {
      if (this.statsInterval) return;
      this.statsInterval = setInterval(async () => {
        if (!this.pc || this.pc.connectionState === 'closed') return;
        const stats = await this.pc.getStats().catch(() => null);
        if (!stats) return;
        for (const report of stats.values()) {
          if (report.type === 'outbound-rtp' && report.kind === 'video') {
            console.log(
              '[Feel Bridge] outbound-rtp video',
              'codecId=' + (report.codecId || ''),
              'bytesSent=' + (report.bytesSent ?? 0),
              'packetsSent=' + (report.packetsSent ?? 0),
              'framesEncoded=' + (report.framesEncoded ?? ''),
            );
            return;
          }
        }
      }, 1000);
    }

    async _ensureGatewayWs() {
      if (this.target?.targetKind !== 'gateway') return;
      if (this.gatewayWs && this.gatewayWs.readyState === WebSocket.OPEN) return;

      const wsUrl = this.target?.wsUrl || BRIDGE_CONSTANTS?.GATEWAY_WS_URL;
      if (!wsUrl) return;

      const ws = new WebSocket(wsUrl);
      this.gatewayWs = ws;
      try {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error(t('errors_gateway_unavailable', [wsUrl]))), 800);
          ws.onopen = () => {
            clearTimeout(timeout);
            resolve();
          };
          ws.onerror = () => {
            clearTimeout(timeout);
            reject(new Error(t('errors_gateway_unavailable', [wsUrl])));
          };
        });
      } catch {
        // Fall back to background signaling path.
        try {
          ws.close();
        } catch {
          // ignore
        }
        this.gatewayWs = null;
        return;
      }

      try {
        ws.send(
          JSON.stringify({
            type: 'hello',
            client: 'f8chromeext-content',
            ts: Date.now(),
          }),
        );
      } catch {
        // ignore
      }

      ws.onmessage = (event) => {
        if (!event?.data) return;
        let msg = null;
        try {
          msg = JSON.parse(String(event.data));
        } catch {
          return;
        }
        const type = msg?.type || null;
        const payload = msg?.payload && typeof msg.payload === 'object' ? msg.payload : msg;
        if (!type) return;
        const sid = payload?.sessionId || msg?.sessionId || null;
        if (!sid || sid !== this.gatewaySessionId) return;

        if (type === 'webrtc.answer') {
          this.handleAnswer(payload?.description ?? null);
        } else if (type === 'webrtc.ice') {
          this.handleRemoteIce(payload?.candidate ?? null);
        } else if (type === 'webrtc.stop') {
          this.stop(payload?.reason || 'stopped', false);
        }
      };
    }

    async start() {
      if (this.target?.targetKind === 'gateway') {
        try {
          this.keepalivePort = chrome.runtime.connect({ name: 'gateway-stream' });
        } catch {
          // ignore
        }
        await this._ensureGatewayWs();
      }
      this.stream = captureVideoStream(this.video);
      if (!this.stream) {
        throw new Error(t('errors_video_uncapturable'));
      }
      this.stream.getTracks().forEach((track) => this.pc.addTrack(track, this.stream));
      if (this.target?.targetKind === 'gateway') {
        this._applyVideoCodecPreferences(GATEWAY_VIDEO_CODEC);
        const videoTrack = this.stream?.getVideoTracks?.()?.[0] ?? null;
        if (videoTrack && typeof videoTrack.applyConstraints === 'function') {
          const idealWidth = Number(this.video?.videoWidth || 0);
          const idealHeight = Number(this.video?.videoHeight || 0);
          const constraints = {
            frameRate: { ideal: 30, max: 30 },
          };
          if (idealWidth > 0 && idealHeight > 0) {
            constraints.width = { ideal: idealWidth };
            constraints.height = { ideal: idealHeight };
          }
          try {
            try {
              videoTrack.contentHint = 'motion';
            } catch {
              // ignore
            }
            await videoTrack.applyConstraints(constraints);
            const settings = typeof videoTrack.getSettings === 'function' ? videoTrack.getSettings() : {};
            console.log('[Feel Bridge] videoTrack constraints applied', constraints, settings);
          } catch (error) {
            console.warn('[Feel Bridge] videoTrack applyConstraints failed', error);
          }
        }
        try {
          const sender = this.pc
            ?.getSenders?.()
            ?.find((s) => s?.track?.kind === 'video' && typeof s?.getParameters === 'function');
          if (sender && typeof sender.setParameters === 'function') {
            const params = sender.getParameters();
            params.encodings = Array.isArray(params.encodings) && params.encodings.length ? params.encodings : [{}];
            params.encodings[0] = {
              ...params.encodings[0],
              maxFramerate: 30,
              maxBitrate: 2_500_000,
            };
            await sender.setParameters(params);
          }
        } catch (error) {
          console.warn('[Feel Bridge] sender.setParameters failed', error);
        }
        this._startStatsLog();
      }
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
      if (this.target?.targetKind === 'gateway' && this.gatewayWs?.readyState === WebSocket.OPEN) {
        this.gatewayWs.send(
          JSON.stringify({
            type: 'webrtc.offer',
            sessionId: this.gatewaySessionId,
            description: descriptionJson,
            video: this.metadata,
            source: {
              title: document.title,
              url: window.location.href,
            },
            ts: Date.now(),
          }),
        );
      } else {
        const resp = await sendRuntimeMessage({
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
        if (resp && resp.ok === false) {
          throw new Error(resp.error || t('errors_gateway_unavailable', [BRIDGE_CONSTANTS.GATEWAY_WS_URL]));
        }
      }
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
      if (this.statsInterval) {
        clearInterval(this.statsInterval);
        this.statsInterval = null;
      }
      if (this.keepalivePort) {
        try {
          this.keepalivePort.disconnect();
        } catch {
          // ignore
        }
        this.keepalivePort = null;
      }
      if (this.gatewayWs) {
        if (notify && this.gatewayWs.readyState === WebSocket.OPEN && this.gatewaySessionId) {
          try {
            this.gatewayWs.send(
              JSON.stringify({
                type: 'webrtc.stop',
                sessionId: this.gatewaySessionId,
                reason,
                ts: Date.now(),
              }),
            );
          } catch {
            // ignore
          }
        }
        try {
          this.gatewayWs.close();
        } catch {
          // ignore
        }
        this.gatewayWs = null;
      }
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
