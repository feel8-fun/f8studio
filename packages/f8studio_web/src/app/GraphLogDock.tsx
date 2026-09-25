import { ChevronDown, ChevronUp, Maximize2, ScrollText } from 'lucide-react';
import { useRef, useState, type CSSProperties, type ReactNode } from 'react';

import { LogsWorkspace } from '../logs/LogsWorkspace';

const HEIGHT_KEY = 'f8studio.graphLogHeight';
const COLLAPSED_KEY = 'f8studio.graphLogCollapsed';
const HEADER_HEIGHT = 30;
const MIN_HEIGHT = 110;
const MAX_HEIGHT = 480;
const DEFAULT_HEIGHT = 200;
const MIN_GRAPH_HEIGHT = 180;

function savedHeight(): number {
  const value = Number(localStorage.getItem(HEIGHT_KEY));
  return Number.isFinite(value) && value >= MIN_HEIGHT
    ? Math.min(MAX_HEIGHT, value) : DEFAULT_HEIGHT;
}

export function GraphLogDock({ children, onOpenLogs }: {
  readonly children: ReactNode;
  readonly onOpenLogs: () => void;
}) {
  const [height, setHeight] = useState(savedHeight);
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(COLLAPSED_KEY) === 'true');
  const layoutRef = useRef<HTMLDivElement>(null);
  const toolbarRef = useRef<HTMLDivElement>(null);
  const resizingRef = useRef(false);
  const dragHeightRef = useRef(height);

  const maxHeight = (): number => Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT,
    (layoutRef.current?.getBoundingClientRect().height ?? MAX_HEIGHT + MIN_GRAPH_HEIGHT) - MIN_GRAPH_HEIGHT));
  const setDockHeight = (next: number): void => {
    const bounded = Math.max(MIN_HEIGHT, Math.min(maxHeight(), next));
    dragHeightRef.current = bounded;
    setHeight(bounded);
    localStorage.setItem(HEIGHT_KEY, String(bounded));
  };
  const setDockCollapsed = (next: boolean): void => {
    setCollapsed(next);
    localStorage.setItem(COLLAPSED_KEY, String(next));
  };
  const finishResize = (): void => {
    if (!resizingRef.current) return;
    resizingRef.current = false;
    if (dragHeightRef.current < MIN_HEIGHT) {
      setDockCollapsed(true);
    } else {
      setDockHeight(dragHeightRef.current);
    }
  };

  return <div className="graph-log-layout" ref={layoutRef}
    style={{ '--graph-log-height': `${collapsed ? HEADER_HEIGHT : height}px` } as CSSProperties}>
    <div className="graph-main">{children}</div>
    <section className={`graph-log-dock${collapsed ? ' graph-log-dock-collapsed' : ''}`} aria-label="Quick logs">
      {!collapsed && <div className="graph-log-resizer" role="separator" aria-label="Resize quick logs" aria-orientation="horizontal"
        aria-valuemin={MIN_HEIGHT} aria-valuemax={maxHeight()} aria-valuenow={height} tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          resizingRef.current = true;
          dragHeightRef.current = height;
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!resizingRef.current || layoutRef.current === null) return;
          const next = layoutRef.current.getBoundingClientRect().bottom - event.clientY;
          dragHeightRef.current = next < MIN_HEIGHT ? HEADER_HEIGHT : Math.min(maxHeight(), next);
          layoutRef.current.style.setProperty('--graph-log-height', `${dragHeightRef.current}px`);
        }}
        onPointerUp={(event) => {
          finishResize();
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={finishResize}
        onLostPointerCapture={finishResize}
        onKeyDown={(event) => {
          let next = height;
          if (event.key === 'ArrowUp') next += 24;
          else if (event.key === 'ArrowDown') next -= 24;
          else if (event.key === 'Home') next = MIN_HEIGHT;
          else if (event.key === 'End') next = maxHeight();
          else return;
          event.preventDefault();
          event.stopPropagation();
          setDockHeight(next);
        }} />}
      <div className="graph-log-header">
        <button type="button" className="graph-log-toggle" aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand quick logs' : 'Collapse quick logs'}
          title={collapsed ? 'Expand quick logs' : 'Collapse quick logs'} onClick={() => setDockCollapsed(!collapsed)}>
          {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          <ScrollText size={14} /><span>Logs</span>
        </button>
        <div className="graph-log-controls" ref={toolbarRef} />
        <button type="button" className="graph-log-open" aria-label="Open log center" title="Open log center" onClick={onOpenLogs}><Maximize2 size={14} /></button>
      </div>
      {!collapsed && <LogsWorkspace compact toolbarTarget={toolbarRef} />}
    </section>
  </div>;
}
