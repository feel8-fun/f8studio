import { Activity, ArrowDown, ArrowUp, CircleDot, Pin, Trash2 } from 'lucide-react';
import { Suspense, useEffect, useMemo, useRef, useState } from 'react';

import type { JsonValue } from '../api/contracts';
import { extensionRendererById, extensionToolById, studioExtensions } from '../extensions/registry';
import { SkeletonOutputPreview } from '../three/SkeletonOutputPreview';
import {
  type PresentationOutput,
  usePresentationConnected,
  usePresentationOutputs,
} from './PresentationStore';
import { PresentationVideo } from './PresentationVideo';

const PINNED_OUTPUTS_KEY = 'f8studio.pinnedOutputs';

function readPinnedOutputs(): readonly string[] {
  const stored = localStorage.getItem(PINNED_OUTPUTS_KEY);
  if (stored === null) return [];
  try {
    const value: unknown = JSON.parse(stored);
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  } catch (error) {
    console.error('Failed to read pinned outputs', error);
    return [];
  }
}

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

export function PresentationWorkspace({ nodeId = null }: { readonly nodeId?: string | null }) {
  const outputs = usePresentationOutputs();
  const [dismissed, setDismissed] = useState<ReadonlyMap<string, PresentationOutput>>(new Map());
  const [pinned, setPinned] = useState<readonly string[]>(readPinnedOutputs);
  const [tab, setTab] = useState<string>('live');
  const visibleOutputs = useMemo(() => {
    const available = [...outputs.values()].filter((output) =>
      (nodeId === null || output.nodeId === nodeId) && dismissed.get(output.nodeId) !== output);
    if (tab !== 'pinned') return available;
    const byId = new Map(available.map((output) => [output.nodeId, output]));
    return pinned.flatMap((id) => { const output = byId.get(id); return output === undefined ? [] : [output]; });
  }, [dismissed, outputs, nodeId, pinned, tab]);
  const connected = usePresentationConnected();
  const activeTool = extensionToolById(tab);
  const ActiveToolComponent = activeTool?.component;
  const extensionTools = studioExtensions.flatMap((extension) => extension.tools ?? []);

  useEffect(() => {
    localStorage.setItem(PINNED_OUTPUTS_KEY, JSON.stringify(pinned));
  }, [pinned]);

  const togglePin = (id: string) => setPinned((current) => current.includes(id)
    ? current.filter((item) => item !== id) : [...current, id]);
  const movePin = (id: string, delta: number) => setPinned((current) => {
    const index = current.indexOf(id);
    const nextIndex = index + delta;
    if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
    const moved = [...current];
    moved.splice(index, 1);
    moved.splice(nextIndex, 0, id);
    return moved;
  });

  return <section className={`presentation-workspace ${nodeId !== null ? 'presentation-workspace-focused' : ''}`} aria-label="Presentation outputs">
    <div className="tool-strip">
      {nodeId === null && <div className="segment" role="tablist" aria-label="Output tools">
        <button className={tab === 'live' ? 'selected' : ''} role="tab" aria-selected={tab === 'live'} onClick={() => setTab('live')}><Activity size={15} />Live outputs</button>
        <button className={tab === 'pinned' ? 'selected' : ''} role="tab" aria-selected={tab === 'pinned'} onClick={() => setTab('pinned')}><Pin size={15} />Pinned</button>
        {extensionTools.map((tool) => <button key={tool.id} className={tab === tool.id ? 'selected' : ''} role="tab" aria-selected={tab === tool.id} onClick={() => setTab(tool.id)}><tool.icon size={15} />{tool.label}</button>)}
      </div>}
      {nodeId !== null && <strong className="focused-output-label">{nodeId}</strong>}
      <span className={`stream-state ${connected ? 'online' : ''}`}><CircleDot size={13} />{connected ? 'Event stream online' : 'Reconnecting'}</span>
      {tab === 'live' && nodeId === null && <button className="icon-button bordered" type="button" aria-label="Clear outputs" title="Clear outputs" onClick={() => setDismissed(new Map(outputs))}><Trash2 size={16} /></button>}
    </div>
    {ActiveToolComponent !== undefined ? <Suspense fallback={<div className="view-loading" role="status">Loading tool</div>}><ActiveToolComponent /></Suspense> : <div className="output-grid">
      {visibleOutputs.map((output) => {
        const ExtensionComponent = extensionRendererById(output.renderer)?.component;
        return <article className="output-panel" key={output.nodeId}>
        <header><span>{output.nodeId}</span><div className="output-panel-actions"><small>{output.renderer}</small>
          {tab === 'pinned' && <>
            <button className="icon-button" type="button" aria-label={`Move ${output.nodeId} up`} title="Move up" disabled={pinned.indexOf(output.nodeId) <= 0} onClick={() => movePin(output.nodeId, -1)}><ArrowUp size={14} /></button>
            <button className="icon-button" type="button" aria-label={`Move ${output.nodeId} down`} title="Move down" disabled={pinned.indexOf(output.nodeId) >= pinned.length - 1} onClick={() => movePin(output.nodeId, 1)}><ArrowDown size={14} /></button>
          </>}
          <button className={`icon-button ${pinned.includes(output.nodeId) ? 'output-pinned' : ''}`} type="button"
            aria-label={`${pinned.includes(output.nodeId) ? 'Unpin' : 'Pin'} ${output.nodeId}`} title={pinned.includes(output.nodeId) ? 'Unpin output' : 'Pin output'}
            onClick={() => togglePin(output.nodeId)}><Pin size={14} /></button>
        </div></header>
        {output.renderer === 'text' && <pre>{JSON.stringify(output.payload.value, null, 2)}</pre>}
        {output.renderer === 'wave' && <WaveCanvas payload={output.payload} />}
        {output.renderer === 'track' && <TrackCanvas payload={output.payload} />}
        {ExtensionComponent !== undefined && <Suspense fallback={<div className="view-loading" role="status">Loading output</div>}><ExtensionComponent payload={output.payload} /></Suspense>}
        {output.renderer === 'video' && <PresentationVideo payload={output.payload} />}
        {output.renderer === 'three_d' && <SkeletonOutputPreview nodeId={output.nodeId} className="output-three" />}
      </article>;})}
      {visibleOutputs.length === 0 && <div className="empty-state centered">{nodeId !== null ? `Waiting for ${nodeId}` : tab === 'pinned' ? 'No pinned outputs are live' : 'Deploy visualization nodes to see live outputs'}</div>}
    </div>}
  </section>;
}
