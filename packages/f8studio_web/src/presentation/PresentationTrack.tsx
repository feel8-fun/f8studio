import { useEffect, useRef } from 'react';

import type { JsonValue } from '../api/contracts';

export function PresentationTrack({ payload, compact = false }: {
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly compact?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext('2d');
    if (canvas === null || canvas === undefined || context === null || context === undefined) return;
    const width = typeof payload.width === 'number' && payload.width > 0 ? payload.width : 1;
    const height = typeof payload.height === 'number' && payload.height > 0 ? payload.height : 1;
    context.fillStyle = '#0a0d10'; context.fillRect(0, 0, canvas.width, canvas.height);
    const tracks = Array.isArray(payload.tracks) ? payload.tracks : [];
    tracks.forEach((track, index) => {
      if (typeof track !== 'object' || track === null || Array.isArray(track)) return;
      const history = Array.isArray(track.history) ? track.history : [];
      const sample = history[history.length - 1];
      if (typeof sample !== 'object' || sample === null || Array.isArray(sample) || !Array.isArray(sample.bbox) || sample.bbox.length < 4) return;
      const [x, y, w, h] = sample.bbox.map(Number);
      if ([x, y, w, h].some((value) => !Number.isFinite(value))) return;
      context.strokeStyle = ['#65c99e', '#62a9e8', '#e5b95c'][index % 3] ?? '#65c99e'; context.lineWidth = 2;
      context.strokeRect((x ?? 0) / width * canvas.width, (y ?? 0) / height * canvas.height, (w ?? 0) / width * canvas.width, (h ?? 0) / height * canvas.height);
      context.fillStyle = context.strokeStyle; context.fillText(String(track.id ?? index), (x ?? 0) / width * canvas.width + 4, (y ?? 0) / height * canvas.height + 14);
    });
  }, [payload]);
  return <canvas className={compact ? 'inline-track-canvas' : 'output-canvas'} ref={ref}
    width={compact ? 480 : 720} height={compact ? 270 : 360} aria-label="Track visualization" />;
}
