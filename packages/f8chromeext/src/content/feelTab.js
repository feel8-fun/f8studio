(async () => {
  const [{ BRIDGE_CONSTANTS }, { MESSAGE_TYPES }, runtime] = await Promise.all([
    import(chrome.runtime.getURL('src/shared/constants.js')),
    import(chrome.runtime.getURL('src/shared/messages.js')),
    import(chrome.runtime.getURL('src/shared/runtime.js')),
  ]);

  const { safeSendMessage } = runtime;
  const { FEEL_META_NAME, ICE_CONFIG, REGISTER_INTERVAL_MS } = BRIDGE_CONSTANTS;

  const isFeelPlayer = () => !!document.querySelector(`meta[name="${FEEL_META_NAME}"]`);

  const dispatchEvent = (name, detail) => {
    window.dispatchEvent(new CustomEvent(name, { detail }));
  };

  class RegistrationScheduler {
    constructor(callback) {
      this.callback = callback;
      this.interval = null;
    }

    start() {
      if (this.interval) return;
      this.interval = setInterval(() => this.callback(), REGISTER_INTERVAL_MS);
    }

    stop() {
      if (!this.interval) return;
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  class ReceiverPeer {
    constructor(sourceInfo) {
      this.source = sourceInfo || null;
      this.pc = new RTCPeerConnection(ICE_CONFIG);
      this.pc.onicecandidate = (event) => {
        if (event.candidate) {
          safeSendMessage({
            type: MESSAGE_TYPES.WEBRTC_ICE,
            payload: { candidate: event.candidate },
          });
        }
      };
      this.pc.ontrack = (event) => {
        const [stream] = event.streams || [];
        if (stream) {
          const video = this.attachStreamToPlayer(stream);
          if (video) {
            video
              .play()
              .then(() => {
                console.info('[Feel Bridge] Remote playback started successfully.');
                dispatchEvent('feel-bridge:remote-stream', { source: this.source });
              })
              .catch((error) => {
                console.warn('[Feel Bridge] Remote playback blocked in Feel tab:', error);
                dispatchEvent('feel-bridge:remote-stream', { source: this.source });
              });
          } else {
            dispatchEvent('feel-bridge:remote-stream', { source: this.source });
          }
        }
      };
    }

    attachStreamToPlayer(stream) {
      const video = document.querySelector('[data-feel-player-video="true"]');
      if (!video) {
        console.warn('[Feel Bridge] Unable to find Feel player video element.');
        return null;
      }
      video.__feelBridgeStream = stream;
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      return video;
    }

    async handleOffer(description) {
      await this.pc.setRemoteDescription(new RTCSessionDescription(description));
      const answer = await this.pc.createAnswer();
      await this.pc.setLocalDescription(answer);
      safeSendMessage({
        type: MESSAGE_TYPES.WEBRTC_ANSWER,
        payload: { description: this.pc.localDescription },
      });
    }

    async handleRemoteIce(candidate) {
      if (!candidate) return;
      try {
        await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
      } catch (error) {
        console.warn('[Feel Bridge] Receiver failed to add ICE candidate', error);
      }
    }

    stop(reason = 'stopped') {
      try {
        this.pc.close();
      } catch {
        // ignore
      }
      const video = document.querySelector('[data-feel-player-video="true"]');
      if (video && video.__feelBridgeStream) {
        video.pause();
        video.srcObject = null;
        delete video.__feelBridgeStream;
      }
      dispatchEvent('feel-bridge:remote-ended', { reason });
    }
  }

  class FeelPlayerBridge {
    constructor() {
      this.receiver = null;
      this.registration = new RegistrationScheduler(() => this.register());
      this.boundOnMessage = (message) => this.handleMessage(message);
      this.boundOnUnload = () => this.unregister();
    }

    register() {
      safeSendMessage({
        type: MESSAGE_TYPES.REGISTER_PLAYER,
        payload: {
          title: document.title,
          url: window.location.href,
        },
      });
    }

    unregister() {
      safeSendMessage({ type: MESSAGE_TYPES.UNREGISTER_PLAYER });
      this.registration.stop();
    }

    start() {
      this.register();
      this.registration.start();
      window.addEventListener('beforeunload', this.boundOnUnload);
      window.addEventListener('unload', this.boundOnUnload);
      chrome.runtime?.onMessage.addListener(this.boundOnMessage);
    }

    stopReceiver(reason = 'stopped') {
      if (this.receiver) {
        this.receiver.stop(reason);
        this.receiver = null;
      }
    }

    stop(reason = 'stopped') {
      this.registration.stop();
      window.removeEventListener('beforeunload', this.boundOnUnload);
      window.removeEventListener('unload', this.boundOnUnload);
      chrome.runtime?.onMessage.removeListener(this.boundOnMessage);
      this.stopReceiver(reason);
      this.unregister();
    }

    async handleOffer(payload) {
      if (!payload?.description) return;
      this.stopReceiver('replaced');
      this.receiver = new ReceiverPeer(payload.source || null);
      await this.receiver.handleOffer(payload.description);
    }

    handleMessage(message) {
      if (!message || typeof message !== 'object') return;
      switch (message.type) {
        case MESSAGE_TYPES.MEDIA_LIST:
          dispatchEvent('feel-bridge:media-list', message.payload);
          break;
        case MESSAGE_TYPES.WEBRTC_OFFER:
          this.handleOffer(message.payload).catch((error) => {
            console.error('[Feel Bridge] Failed to handle WebRTC offer', error);
            this.stopReceiver('error');
          });
          break;
        case MESSAGE_TYPES.WEBRTC_ICE:
          if (message.payload?.direction === 'sender') {
            this.receiver?.handleRemoteIce(message.payload?.candidate);
          }
          break;
        case MESSAGE_TYPES.WEBRTC_STOP:
          this.stopReceiver(message.payload?.reason || 'stopped');
          break;
        default:
          break;
      }
    }
  }

  if (isFeelPlayer()) {
    const bridge = new FeelPlayerBridge();
    bridge.start();
  }
})();
