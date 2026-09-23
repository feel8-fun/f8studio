import { Activity, CircleDot, Gauge, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import type { JsonValue } from '../api/contracts';
import {
  type PresentationOutput,
  usePresentationConnected,
  usePresentationOutputs,
} from './PresentationStore';
import { PresentationVideo } from './PresentationVideo';
import { TemplateCapture } from './TemplateCapture';

function WaveCanvas({ payload }: { readonly payload: Readonly<Record<string, JsonValue>> }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext('2d');
    if (canvas === null || canvas === undefined || context === null || context === undefined) return;
    const seriesValue = payload.series;
    const series = typeof seriesValue === 'object' && seriesValue !== null && !Array.isArray(seriesValue) ? seriesValue : {};
    const allPoints = Object.values(series).flatMap((value) => Array.isArray(value) ? value : []).filter((value): value is readonly JsonValue[] => Array.isArray(value) && typeof value[0] === 'number' && typeof value[1] === 'number');
    const values = allPoints.map((point) => Number(point[1]));
    const times = allPoints.map((point) => Number(point[0]));
    const minY = typeof payload.minVal === 'number' ? payload.minVal : Math.min(...values, 0);
    const maxY = typeof payload.maxVal === 'number' ? payload.maxVal : Math.max(...values, 1);
    const minX = Math.min(...times, Date.now() - 10_000);
    const maxX = Math.max(...times, Date.now());
    context.fillStyle = '#0a0d10'; context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#262c33'; context.lineWidth = 1;
    for (let i = 1; i < 4; i += 1) { const y = i * canvas.height / 4; context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke(); }
    const colors = ['#65c99e', '#62a9e8', '#e5b95c', '#dc7084'];
    Object.entries(series).forEach(([key, raw], seriesIndex) => {
      if (!Array.isArray(raw)) return;
      context.strokeStyle = colors[seriesIndex % colors.length] ?? '#65c99e'; context.lineWidth = 2; context.beginPath();
      let started = false;
      raw.forEach((point) => {
        if (!Array.isArray(point) || typeof point[0] !== 'number' || typeof point[1] !== 'number') return;
        const x = (point[0] - minX) / Math.max(1, maxX - minX) * canvas.width;
        const y = canvas.height - (point[1] - minY) / Math.max(1e-9, maxY - minY) * canvas.height;
        if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
      });
      context.stroke();
      if (payload.showLegend === true) { context.fillStyle = context.strokeStyle; context.fillText(key, 10, 18 + seriesIndex * 16); }
    });
  }, [payload]);
  return <canvas className="output-canvas" ref={ref} width={720} height={240} aria-label="Curve renderer" />;
}

function TrackCanvas({ payload }: { readonly payload: Readonly<Record<string, JsonValue>> }) {
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
  return <canvas className="output-canvas" ref={ref} width={720} height={360} aria-label="Track renderer" />;
}

function TCodeView({ payload }: { readonly payload: Readonly<Record<string, JsonValue>> }) {
  const line = typeof payload.line === 'string' ? payload.line : '';
  const values = useMemo(() => {
    const result = new Map<string, number>();
    for (const match of line.matchAll(/([A-Z]\d)(\d{1,5})/g)) result.set(match[1] ?? '', Math.max(0, Math.min(9999, Number(match[2]))));
    return result;
  }, [line]);
  return <div className="tcode-view"><code>{line || 'Waiting for TCode'}</code><div className="tcode-channels">{[...values.entries()].map(([channel, value]) => <label key={channel}><span>{channel}</span><meter min={0} max={9999} value={value} /><output>{value}</output></label>)}</div></div>;
}

export function PresentationWorkspace() {
  const outputs = usePresentationOutputs();
  const [dismissed, setDismissed] = useState<ReadonlyMap<string, PresentationOutput>>(new Map());
  const visibleOutputs = useMemo(() => [...outputs.values()].filter((output) =>
    output.renderer !== 'three_d' && dismissed.get(output.nodeId) !== output), [dismissed, outputs]);
  const connected = usePresentationConnected();
  const [tab, setTab] = useState<'live' | 'template'>('live');

  return <section className="presentation-workspace" aria-label="Presentation outputs">
    <div className="tool-strip">
      <div className="segment" role="tablist" aria-label="Output tools">
        <button className={tab === 'live' ? 'selected' : ''} role="tab" aria-selected={tab === 'live'} onClick={() => setTab('live')}><Activity size={15} />Live outputs</button>
        <button className={tab === 'template' ? 'selected' : ''} role="tab" aria-selected={tab === 'template'} onClick={() => setTab('template')}><Gauge size={15} />Template</button>
      </div>
      <span className={`stream-state ${connected ? 'online' : ''}`}><CircleDot size={13} />{connected ? 'Event stream online' : 'Reconnecting'}</span>
      {tab === 'live' && <button className="icon-button bordered" type="button" aria-label="Clear outputs" title="Clear outputs" onClick={() => setDismissed(new Map(outputs))}><Trash2 size={16} /></button>}
    </div>
    {tab === 'template' ? <TemplateCapture /> : <div className="output-grid">
      {visibleOutputs.map((output) => <article className="output-panel" key={output.nodeId}>
        <header><span>{output.nodeId}</span><small>{output.renderer}</small></header>
        {output.renderer === 'text' && <pre>{JSON.stringify(output.payload.value, null, 2)}</pre>}
        {output.renderer === 'wave' && <WaveCanvas payload={output.payload} />}
        {output.renderer === 'track' && <TrackCanvas payload={output.payload} />}
        {output.renderer === 'tcode' && <TCodeView payload={output.payload} />}
        {output.renderer === 'video' && <PresentationVideo payload={output.payload} />}
      </article>)}
      {visibleOutputs.length === 0 && <div className="empty-state centered">Deploy visualization nodes to see live outputs</div>}
    </div>}
  </section>;
}
