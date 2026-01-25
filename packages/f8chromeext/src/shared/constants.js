export const BRIDGE_CONSTANTS = {
  FEEL_META_NAME: 'feel-bridge-target',
  REGISTER_INTERVAL_MS: 15_000,
  DEFAULT_SIGNALING_TARGET: 'auto',
  GATEWAY_WS_URL: 'ws://127.0.0.1:8765/ws',
  ICE_CONFIG: {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun.cloudflare.com:3478' },
      { urls: 'stun:global.stun.twilio.com:3478' },
    ],
  },
  HIGHLIGHT_CLASS: 'feel-bridge-highlight',
};

export const FEEL_SITE_URL = 'https://dance.feel8.fun/';
