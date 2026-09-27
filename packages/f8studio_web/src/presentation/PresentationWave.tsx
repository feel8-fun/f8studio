import { useEffect, useRef } from 'react';

import type { JsonValue } from '../api/contracts';

interface Point { readonly time: number; readonly value: number }

function isJsonRecord(value: JsonValue | undefined): value is Readonly<Record<string, JsonValue>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function waveSeries(value: JsonValue | undefined): Readonly<Record<string, readonly Point[]>> {
  if (!isJsonRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([name, raw]) => [
    name,
    Array.isArray(raw) ? raw.flatMap((item): Point[] => {
      if (!Array.isArray(item) || typeof item[0] !== 'number' || typeof item[1] !== 'number' ||
        !Number.isFinite(item[0]) || !Number.isFinite(item[1])) return [];
      return [{ time: item[0], value: item[1] }];
    }) : [],
  ]));
}

function seriesColor(colors: JsonValue | undefined, name: string, index: number): string {
  if (isJsonRecord(colors)) {
    const rgb = colors[name];
    if (Array.isArray(rgb) && rgb.length === 3 && rgb.every((part) => typeof part === 'number' && Number.isFinite(part))) {
      const channels = rgb.map((part) => Math.max(0, Math.min(255, Math.round(part))));
      return `rgb(${channels.join(', ')})`;
    }
  }
  return ['#65c99e', '#62a9e8', '#e5b95c', '#dc7084'][index % 4] ?? '#65c99e';
}

export function PresentationWave({ payload, compact = false }: {
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly compact?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext('2d');
    if (canvas === null || canvas === undefined || context === null || context === undefined) return;
    const series = waveSeries(payload.series);
    const windowMs = typeof payload.windowMs === 'number' && payload.windowMs > 0 ? payload.windowMs : 10_000;
    const nowMs = typeof payload.nowMs === 'number' && Number.isFinite(payload.nowMs) ? payload.nowMs : Date.now();
    const minX = nowMs - windowMs;
    const visible = Object.entries(series).map(([name, points]) => [name, points.filter((point) => point.time >= minX && point.time <= nowMs)] as const);
    let observedMin = 0;
    let observedMax = 1;
    for (const [, points] of visible) {
      for (const point of points) {
        observedMin = Math.min(observedMin, point.value);
        observedMax = Math.max(observedMax, point.value);
      }
    }
    let minY = typeof payload.minVal === 'number' && Number.isFinite(payload.minVal) ? payload.minVal : observedMin;
    let maxY = typeof payload.maxVal === 'number' && Number.isFinite(payload.maxVal) ? payload.maxVal : observedMax;
    if (maxY <= minY) { minY -= 0.5; maxY = minY + 1; }
    const { width, height } = canvas;
    context.fillStyle = '#0a0d10';
    context.fillRect(0, 0, width, height);
    context.strokeStyle = '#262c33';
    context.lineWidth = 1;
    for (let i = 1; i < 4; i += 1) {
      const y = i * height / 4;
      context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
    }
    context.save();
    context.beginPath(); context.rect(0, 0, width, height); context.clip();
    visible.forEach(([name, points], index) => {
      if (points.length === 0) return;
      context.strokeStyle = seriesColor(payload.colors, name, index);
      context.lineWidth = 2;
      context.beginPath();
      points.forEach((point, pointIndex) => {
        const x = (point.time - minX) / windowMs * width;
        const y = height - (point.value - minY) / (maxY - minY) * height;
        if (pointIndex === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
    });
    context.restore();
    if (payload.showLegend === true) visible.forEach(([name], index) => {
      context.fillStyle = seriesColor(payload.colors, name, index);
      context.fillText(name, 8, 16 + index * 15);
    });
  }, [payload]);
  return <canvas className={compact ? 'inline-wave-canvas' : 'output-canvas'} ref={ref}
    width={compact ? 480 : 720} height={compact ? 270 : 240} aria-label="Wave visualization" />;
}
