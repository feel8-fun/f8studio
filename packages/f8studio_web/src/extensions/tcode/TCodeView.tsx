import { lazy, Suspense } from 'react';
import type { OSRModel } from 'osr-emu';
import { Box, ChartNoAxesColumn } from 'lucide-react';

import type { JsonValue } from '../../api/contracts';
import { mergeTCodeChannels } from './tcodeChannels';
import { setTCodeViewMode, useTCodeViewMode } from './tcodeViewMode';

const TCode3D = lazy(() => import('./TCode3D').then((module) => ({ default: module.TCode3D })));

function tcodeModel(value: JsonValue | undefined): OSRModel {
  return value === 'OSR2' || value === 'SSR1' ? value : 'SR6';
}

export function TCodeView({ nodeId, payload, compact = false }: {
  readonly nodeId: string;
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly compact?: boolean;
}) {
  const line = typeof payload.line === 'string' ? payload.line : '';
  const model = tcodeModel(payload.model);
  const resetVersion = typeof payload.resetVersion === 'number' ? payload.resetVersion : 0;
  const mode = useTCodeViewMode(nodeId);
  const values = Object.entries(mergeTCodeChannels(payload.channels, line));
  const bars = <div className={`tcode-channels ${mode === 'bars' ? 'tcode-channels-primary' : ''}`}>
    {values.length === 0 ? <span className="inline-video-placeholder">Waiting for TCode</span> :
      values.map(([channel, value]) => <label key={channel}><span>{channel}</span><meter min={0} max={9999} value={value} /><output>{value}</output></label>)}
  </div>;

  return <div className={`tcode-view ${compact ? 'tcode-view-compact' : ''}`}>
    <div className="tcode-mode-control nodrag nowheel" role="group" aria-label="TCode display mode">
      <button type="button" className={mode === '3d' ? 'selected' : ''} aria-label="Show 3D model" title="Show 3D model" aria-pressed={mode === '3d'} onClick={() => setTCodeViewMode(nodeId, '3d')}><Box size={compact ? 12 : 14} /></button>
      <button type="button" className={mode === 'bars' ? 'selected' : ''} aria-label="Show channel bars" title="Show channel bars" aria-pressed={mode === 'bars'} onClick={() => setTCodeViewMode(nodeId, 'bars')}><ChartNoAxesColumn size={compact ? 12 : 14} /></button>
    </div>
    {mode === 'bars' ? bars : <Suspense fallback={<div className="tcode-stage" role="status">Loading 3D</div>}>
      <TCode3D payload={payload} line={line} model={model} resetVersion={resetVersion} />
    </Suspense>}
    {!compact && <div className="tcode-details">
      <span className="tcode-model">{model}</span>
      <code>{line.trim() || 'Waiting for TCode'}</code>
      {mode === '3d' && bars}
    </div>}
  </div>;
}
