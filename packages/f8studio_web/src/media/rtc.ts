import { fetchRtcConfiguration } from '../api/client';

const ICE_GATHERING_TIMEOUT_MS = 15_000;

export async function createRtcPeerConnection(): Promise<RTCPeerConnection> {
  const configuration = await fetchRtcConfiguration();
  return new RTCPeerConnection({
    iceServers: configuration.iceServers.map((server) => ({
      urls: [...server.urls],
      ...(server.username === undefined ? {} : { username: server.username }),
      ...(server.credential === undefined ? {} : { credential: server.credential }),
    })),
    iceTransportPolicy: configuration.iceTransportPolicy,
  });
}

export async function waitForIceGatheringComplete(peer: RTCPeerConnection): Promise<void> {
  if (peer.iceGatheringState === 'complete') return;
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener('icegatheringstatechange', onStateChange);
      reject(new Error('ICE candidate gathering timed out'));
    }, ICE_GATHERING_TIMEOUT_MS);
    const onStateChange = () => {
      if (peer.iceGatheringState !== 'complete') return;
      window.clearTimeout(timeout);
      peer.removeEventListener('icegatheringstatechange', onStateChange);
      resolve();
    };
    peer.addEventListener('icegatheringstatechange', onStateChange);
    onStateChange();
  });
}
