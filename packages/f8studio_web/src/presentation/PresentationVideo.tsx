import { RefreshCw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import type { JsonValue } from '../api/contracts';
import { videoSessionPool, type VideoSessionLease, type VideoSessionSnapshot } from '../media/VideoSessionPool';

const CONNECTING: VideoSessionSnapshot = { kind: 'connecting', stream: null };

function useVideoSession(source: string): readonly [VideoSessionSnapshot, () => void] {
  const [snapshot, setSnapshot] = useState<VideoSessionSnapshot>(CONNECTING);
  const leaseRef = useRef<VideoSessionLease | null>(null);

  useEffect(() => {
    leaseRef.current = null;
    setSnapshot(CONNECTING);
    if (source === '') return;
    const lease = videoSessionPool.acquire(source, 'thumbnail');
    leaseRef.current = lease;
    const update = () => setSnapshot(lease.getSnapshot());
    const unsubscribe = lease.subscribe(update);
    update();
    return () => {
      unsubscribe();
      lease.release();
      if (leaseRef.current === lease) leaseRef.current = null;
    };
  }, [source]);

  return [snapshot, () => leaseRef.current?.retry()];
}

export function PresentationVideo({
  payload,
  compact = false,
}: {
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly compact?: boolean;
}) {
  const source = typeof payload.videoStreamKey === 'string' ? payload.videoStreamKey.trim() : '';
  const scaleMode = payload.scaleMode === 'native' ? 'native' : 'fit';
  const [snapshot, retry] = useVideoSession(source);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    setDimensions({ width: 0, height: 0 });
  }, [source]);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) return;
    video.srcObject = snapshot.stream;
    return () => {
      if (video.srcObject === snapshot.stream) video.srcObject = null;
    };
  }, [snapshot.stream]);

  const effectiveKind = source === '' ? 'error' : snapshot.kind;
  const errorMessage = source === ''
    ? 'Video input is not connected'
    : snapshot.kind === 'error' ? snapshot.message : '';

  return <div
    className={`presentation-video presentation-video-${scaleMode} ${compact ? 'presentation-video-compact' : ''}`}
    data-video-source={source}
  >
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      onLoadedMetadata={(event) => {
        const video = event.currentTarget;
        setDimensions({ width: video.videoWidth, height: video.videoHeight });
        void video.play().catch((reason: unknown) => {
          console.error('Failed to start presentation video playback', reason);
        });
      }}
    />
    <div className={`presentation-video-status status-${effectiveKind}`} role="status">
      {effectiveKind === 'connecting' && 'Connecting'}
      {effectiveKind === 'playing' && (dimensions.width > 0 ? `${dimensions.width}x${dimensions.height}` : 'Live')}
      {effectiveKind === 'error' && <><span>{errorMessage}</span>{source !== '' && <button type="button" className="icon-button" aria-label="Reconnect video" title="Reconnect video" onClick={retry}><RefreshCw size={15} /></button>}</>}
    </div>
  </div>;
}
