import { CircleStop, Play, Radio } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';

import { closeMediaSession, createMediaSession, fetchMediaSample } from '../api/client';
import type { MediaSample } from '../api/contracts';
import { createRtcPeerConnection, waitForIceGatheringComplete } from './rtc';

type MediaStatus =
  | { readonly kind: 'idle' }
  | { readonly kind: 'connecting' }
  | { readonly kind: 'reconnecting' }
  | { readonly kind: 'playing'; readonly width: number; readonly height: number; readonly fps: number }
  | { readonly kind: 'error'; readonly message: string };

interface NegotiatedVideo {
  readonly width: number;
  readonly height: number;
  readonly fps: number;
}

export function WebRtcVideo() {
  const [source, setSource] = useState('synthetic://bars');
  const [quality, setQuality] = useState<'thumbnail' | 'main'>('thumbnail');
  const [overlay, setOverlay] = useState(false);
  const [status, setStatus] = useState<MediaStatus>({ kind: 'idle' });
  const [sample, setSample] = useState<MediaSample | { readonly error: string } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const sessionRef = useRef<string | null>(null);
  const negotiatedRef = useRef<NegotiatedVideo | null>(null);
  const operationRef = useRef(0);

  const stop = useCallback(async () => {
    operationRef.current += 1;
    const peer = peerRef.current;
    peerRef.current = null;
    peer?.close();
    const sessionId = sessionRef.current;
    sessionRef.current = null;
    negotiatedRef.current = null;
    if (videoRef.current !== null) videoRef.current.srcObject = null;
    setSample(null);
    try {
      if (sessionId !== null) await closeMediaSession(sessionId);
      setStatus({ kind: 'idle' });
    } catch (error: unknown) {
      setStatus({ kind: 'error', message: error instanceof Error ? error.message : 'Media session close failed' });
    }
  }, []);

  useEffect(() => {
    return () => {
      operationRef.current += 1;
      const peer = peerRef.current;
      peerRef.current = null;
      peer?.close();
      const sessionId = sessionRef.current;
      sessionRef.current = null;
      negotiatedRef.current = null;
      if (sessionId !== null) {
        void closeMediaSession(sessionId).catch((error: unknown) => {
          console.error('Failed to release media session during unmount', error);
        });
      }
    };
  }, []);

  const start = useCallback(async () => {
    const operation = operationRef.current + 1;
    operationRef.current = operation;
    const normalizedSource = source.trim();
    if (normalizedSource.length === 0) {
      setStatus({ kind: 'error', message: 'Enter a video source' });
      return;
    }
    setStatus({ kind: 'connecting' });
    let peer: RTCPeerConnection;
    try {
      peer = await createRtcPeerConnection();
      if (operationRef.current !== operation) {
        peer.close();
        return;
      }
    } catch (error: unknown) {
      setStatus({ kind: 'error', message: error instanceof Error ? error.message : 'RTC configuration failed' });
      return;
    }
    let createdSessionId: string | null = null;
    peerRef.current = peer;
    peer.addTransceiver('video', { direction: 'recvonly' });
    peer.ontrack = (event) => {
      if (peerRef.current !== peer) return;
      const video = videoRef.current;
      if (video !== null) video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
    };
    peer.onconnectionstatechange = () => {
      if (peerRef.current !== peer) return;
      if (peer.connectionState === 'disconnected') {
        setStatus({ kind: 'reconnecting' });
      } else if (peer.connectionState === 'connected') {
        const negotiated = negotiatedRef.current;
        if (negotiated === null) return;
        const video = videoRef.current;
        const width = video !== null && video.videoWidth > 0 ? video.videoWidth : negotiated.width;
        const height = video !== null && video.videoHeight > 0 ? video.videoHeight : negotiated.height;
        setStatus({ kind: 'playing', width, height, fps: negotiated.fps });
      } else if (peer.connectionState === 'failed') {
        peerRef.current = null;
        peer.close();
        const failedSessionId = sessionRef.current;
        sessionRef.current = null;
        negotiatedRef.current = null;
        if (videoRef.current !== null) videoRef.current.srcObject = null;
        if (failedSessionId !== null) {
          void closeMediaSession(failedSessionId).catch((error: unknown) => {
            console.error('Failed to release failed media session', error);
          });
        }
        setStatus({
          kind: 'error',
          message: `WebRTC connection failed (ICE: ${peer.iceConnectionState}). Check UDP, VPN, or TURN connectivity.`,
        });
      }
    };
    try {
      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);
      await waitForIceGatheringComplete(peer);
      if (operationRef.current !== operation || peerRef.current !== peer) {
        peer.close();
        return;
      }
      if (peer.localDescription === null) throw new Error('Browser did not create a local description');
      const answer = await createMediaSession(normalizedSource, quality, peer.localDescription, overlay);
      createdSessionId = answer.sessionId;
      if (operationRef.current !== operation || peerRef.current !== peer) {
        peer.close();
        await closeMediaSession(createdSessionId);
        return;
      }
      sessionRef.current = createdSessionId;
      negotiatedRef.current = { width: answer.maxWidth, height: answer.maxHeight, fps: answer.maxFps };
      await peer.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    } catch (error: unknown) {
      peer.close();
      if (peerRef.current === peer) {
        peerRef.current = null;
        sessionRef.current = null;
        negotiatedRef.current = null;
      }
      if (createdSessionId !== null) {
        try {
          await closeMediaSession(createdSessionId);
        } catch (closeError: unknown) {
          console.error('Failed to release media session after negotiation failure', closeError);
        }
      }
      if (operationRef.current === operation) {
        setStatus({ kind: 'error', message: error instanceof Error ? error.message : 'Media negotiation failed' });
      }
    }
  }, [overlay, quality, source]);

  const active = status.kind === 'connecting' || status.kind === 'reconnecting' || status.kind === 'playing';
  const sampleAt = useCallback(async (event: ReactMouseEvent<HTMLVideoElement>) => {
    const video = event.currentTarget;
    if (status.kind !== 'playing' || video.videoWidth <= 0 || video.videoHeight <= 0) return;
    const bounds = video.getBoundingClientRect();
    const scale = Math.min(bounds.width / video.videoWidth, bounds.height / video.videoHeight);
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const offsetX = (bounds.width - renderedWidth) / 2;
    const offsetY = (bounds.height - renderedHeight) / 2;
    const localX = event.clientX - bounds.left - offsetX;
    const localY = event.clientY - bounds.top - offsetY;
    if (localX < 0 || localY < 0 || localX >= renderedWidth || localY >= renderedHeight) return;
    const x = Math.min(video.videoWidth - 1, Math.floor(localX / scale));
    const y = Math.min(video.videoHeight - 1, Math.floor(localY / scale));
    try {
      setSample(await fetchMediaSample(source.trim(), x, y));
    } catch (error: unknown) {
      setSample({ error: error instanceof Error ? error.message : 'Media sample failed' });
    }
  }, [source, status.kind]);

  return (
    <section className="media-workspace" aria-label="WebRTC video">
      <div className="media-toolbar">
        <label className="source-field">
          <Radio size={15} aria-hidden="true" />
          <input value={source} onChange={(event) => setSource(event.target.value)} disabled={active} aria-label="Video source" />
        </label>
        <div className="segment" aria-label="Video quality">
          <button className={quality === 'thumbnail' ? 'selected' : ''} onClick={() => setQuality('thumbnail')} disabled={active}>Thumbnail</button>
          <button className={quality === 'main' ? 'selected' : ''} onClick={() => setQuality('main')} disabled={active}>Main</button>
        </div>
        <label className="overlay-toggle">
          <input type="checkbox" checked={overlay} onChange={(event) => setOverlay(event.target.checked)} disabled={active} />
          <span>Exact overlay</span>
        </label>
        {active ? (
          <button className="command-button" type="button" onClick={() => void stop()}><CircleStop size={16} />Stop</button>
        ) : (
          <button className="command-button primary" type="button" onClick={() => void start()}><Play size={16} />Connect</button>
        )}
      </div>
      <div className="video-stage" data-session-id={sessionRef.current ?? ''}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className={status.kind === 'playing' ? 'sample-enabled' : ''}
          onClick={(event) => void sampleAt(event)}
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            setStatus((current) => current.kind === 'playing' ? {
              ...current,
              width: video.videoWidth,
              height: video.videoHeight,
            } : current);
            void video.play();
          }}
        />
        <div className={`media-status status-${status.kind}`} role="status">
          {status.kind === 'idle' && 'Ready'}
          {status.kind === 'connecting' && 'Negotiating'}
          {status.kind === 'reconnecting' && 'Reconnecting'}
          {status.kind === 'playing' && `${status.width}×${status.height} · ≤${status.fps} fps`}
          {status.kind === 'error' && status.message}
        </div>
        {sample !== null && (
          <div className="media-sample" role="status">
            {'error' in sample ? sample.error : (
              `Latest raw · frame ${sample.frameId} · (${sample.x}, ${sample.y}) · ${JSON.stringify(sample.value)}`
            )}
          </div>
        )}
      </div>
    </section>
  );
}
