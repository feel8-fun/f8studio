import { Activity, ArrowDown, ArrowUp, CircleDot, Pin, Trash2 } from 'lucide-react';
import { Suspense, useEffect, useMemo, useState } from 'react';

import { presentationRendererById, extensionToolById, studioExtensions } from '../extensions/registry';
import {
  type PresentationOutput,
  usePresentationConnected,
  usePresentationOutputs,
} from './PresentationStore';

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
        const renderer = presentationRendererById(output.renderer);
        const OutputComponent = renderer?.component;
        return <article className={`output-panel ${renderer?.panelClass ?? ''}`} key={output.nodeId}>
        <header><span>{output.nodeId}</span><div className="output-panel-actions"><small>{output.renderer}</small>
          {tab === 'pinned' && <>
            <button className="icon-button" type="button" aria-label={`Move ${output.nodeId} up`} title="Move up" disabled={pinned.indexOf(output.nodeId) <= 0} onClick={() => movePin(output.nodeId, -1)}><ArrowUp size={14} /></button>
            <button className="icon-button" type="button" aria-label={`Move ${output.nodeId} down`} title="Move down" disabled={pinned.indexOf(output.nodeId) >= pinned.length - 1} onClick={() => movePin(output.nodeId, 1)}><ArrowDown size={14} /></button>
          </>}
          <button className={`icon-button ${pinned.includes(output.nodeId) ? 'output-pinned' : ''}`} type="button"
            aria-label={`${pinned.includes(output.nodeId) ? 'Unpin' : 'Pin'} ${output.nodeId}`} title={pinned.includes(output.nodeId) ? 'Unpin output' : 'Pin output'}
            onClick={() => togglePin(output.nodeId)}><Pin size={14} /></button>
        </div></header>
        {OutputComponent !== undefined && <Suspense fallback={<div className="view-loading" role="status">Loading output</div>}><OutputComponent nodeId={output.nodeId} payload={output.payload} compact={nodeId === null} /></Suspense>}
      </article>;})}
      {visibleOutputs.length === 0 && <div className="empty-state centered">{nodeId !== null ? `Waiting for ${nodeId}` : tab === 'pinned' ? 'No pinned outputs are live' : 'Deploy visualization nodes to see live outputs'}</div>}
    </div>}
  </section>;
}
