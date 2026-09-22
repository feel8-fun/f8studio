import {
  isAudioSessionAnswer,
  isHealthStatus,
  isMediaSample,
  isMediaSessionAnswer,
  isRtcConfigurationResponse,
  type HealthStatus,
  type AudioSessionAnswer,
  type MediaSample,
  type MediaSessionAnswer,
  type RtcConfigurationResponse,
} from './contracts';

export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  const response = await fetch('/api/health', { signal });
  if (!response.ok) {
    throw new Error(`Health request failed with HTTP ${response.status}`);
  }
  const body: unknown = await response.json();
  if (!isHealthStatus(body)) {
    throw new Error('Health response does not match f8studio-api/1');
  }
  return body;
}

export async function fetchRtcConfiguration(signal?: AbortSignal): Promise<RtcConfigurationResponse> {
  const response = await fetch('/api/media/rtc-configuration', { signal });
  if (!response.ok) throw new Error(`RTC configuration request failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isRtcConfigurationResponse(body)) throw new Error('RTC configuration does not match f8studio-api/1');
  return body;
}

export async function createMediaSession(
  source: string,
  quality: 'thumbnail' | 'main',
  description: RTCSessionDescriptionInit,
  overlay = false,
  signal?: AbortSignal,
): Promise<MediaSessionAnswer> {
  const response = await fetch('/api/media/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, quality, sdp: description.sdp, type: description.type, overlay }),
    signal,
  });
  if (!response.ok) throw new Error(`Media negotiation failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isMediaSessionAnswer(body)) throw new Error('Media answer does not match f8studio-api/1');
  return body;
}

export async function createAudioSession(
  source: string,
  description: RTCSessionDescriptionInit,
  signal?: AbortSignal,
): Promise<AudioSessionAnswer> {
  const response = await fetch('/api/audio/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, sdp: description.sdp, type: description.type }),
    signal,
  });
  if (!response.ok) throw new Error(`Audio negotiation failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isAudioSessionAnswer(body)) throw new Error('Audio answer does not match f8studio-api/1');
  return body;
}

export async function closeAudioSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/audio/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Audio session close failed with HTTP ${response.status}`);
  }
}

export async function closeMediaSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/media/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Media session close failed with HTTP ${response.status}`);
  }
}

export async function fetchMediaSample(source: string, x: number, y: number): Promise<MediaSample> {
  const query = new URLSearchParams({ source, x: String(x), y: String(y) });
  const response = await fetch(`/api/media/sample?${query.toString()}`);
  if (!response.ok) throw new Error(`Media sample failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isMediaSample(body)) throw new Error('Media sample does not match f8studio-api/1');
  return body;
}
