import { Activity, Archive, AudioLines, Boxes, CircleDot, Code2, Cuboid, Plug, Settings2, Video, type LucideIcon } from 'lucide-react';
import { lazy, Suspense, useEffect, useState } from 'react';

import { fetchHealth } from '../api/client';
import type { HealthStatus } from '../api/contracts';
import { GraphWorkspace } from '../graph/GraphWorkspace';
import { PresentationProvider } from '../presentation/PresentationStore';

const WebRtcVideo = lazy(() => import('../media/WebRtcVideo').then((module) => ({ default: module.WebRtcVideo })));
const WebRtcAudio = lazy(() => import('../media/WebRtcAudio').then((module) => ({ default: module.WebRtcAudio })));
const SkeletonViewport = lazy(() => import('../three/SkeletonViewport').then((module) => ({ default: module.SkeletonViewport })));
const AssetsWorkspace = lazy(() => import('../assets/AssetsWorkspace').then((module) => ({ default: module.AssetsWorkspace })));
const CodeWorkspace = lazy(() => import('../editor/CodeWorkspace').then((module) => ({ default: module.CodeWorkspace })));
const PresentationWorkspace = lazy(() => import('../presentation/PresentationWorkspace').then((module) => ({ default: module.PresentationWorkspace })));
const LocalWorkspace = lazy(() => import('../local/LocalWorkspace').then((module) => ({ default: module.LocalWorkspace })));

type WorkspaceView = 'graph' | 'assets' | 'code' | 'outputs' | 'video' | 'audio' | 'three' | 'local';

interface WorkspaceDefinition {
  readonly view: WorkspaceView;
  readonly label: string;
  readonly title: string;
  readonly icon: LucideIcon;
}

const WORKSPACES: readonly WorkspaceDefinition[] = [
  { view: 'graph', label: 'Graph', title: 'Graph Editor', icon: Boxes },
  { view: 'assets', label: 'Assets', title: 'Assets', icon: Archive },
  { view: 'code', label: 'Code', title: 'Code & Schema', icon: Code2 },
  { view: 'outputs', label: 'Outputs', title: 'Live Outputs', icon: Activity },
  { view: 'video', label: 'Video', title: 'Media Lab', icon: Video },
  { view: 'three', label: '3D', title: 'Media Lab', icon: Cuboid },
  { view: 'audio', label: 'Audio', title: 'Media Lab', icon: AudioLines },
  { view: 'local', label: 'Local integrations', title: 'Local Integrations', icon: Plug },
];

type ConnectionState =
  | { readonly kind: 'connecting' }
  | { readonly kind: 'online'; readonly health: HealthStatus }
  | { readonly kind: 'offline'; readonly message: string };

export function App() {
  const [connection, setConnection] = useState<ConnectionState>({ kind: 'connecting' });
  const [view, setView] = useState<WorkspaceView>('graph');

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal).then(
      (health) => setConnection({ kind: 'online', health }),
      (error: unknown) => {
        if (controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : 'Unknown connection error';
        setConnection({ kind: 'offline', message });
      },
    );
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
      const index = Number(event.key) - 1;
      const target = WORKSPACES[index]?.view;
      if (target === undefined) return;
      event.preventDefault();
      setView(target);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const statusText =
    connection.kind === 'connecting'
      ? 'Connecting'
      : connection.kind === 'online'
        ? `Local server ${connection.health.version}`
        : 'Server unavailable';

  return (
    <PresentationProvider><main className="studio-shell">
      <header className="topbar">
        <div className="brand">Feel8 Studio</div>
        <div className={`connection connection-${connection.kind}`} role="status">
          <CircleDot size={14} aria-hidden="true" />
          <span>{statusText}</span>
        </div>
        <button className="icon-button" type="button" aria-label="Settings" title="Settings" disabled>
          <Settings2 size={18} />
        </button>
      </header>

      <aside className="rail" aria-label="Workspace navigation">
        {WORKSPACES.map(({ view: target, label, icon: Icon }) => <button
          className={`rail-button ${view === target ? 'rail-button-active' : ''}`}
          type="button"
          aria-label={label}
          title={label}
          key={target}
          onClick={() => setView(target)}
        ><Icon size={20} /></button>)}
      </aside>

      <section className="workspace" aria-labelledby="workspace-title">
        <div className="workspace-toolbar">
          <h1 id="workspace-title">{WORKSPACES.find((workspace) => workspace.view === view)?.title}</h1>
          {(view === 'video' || view === 'audio' || view === 'three') && <div className="view-tabs" role="tablist" aria-label="Media view">
            <button role="tab" aria-selected={view === 'video'} onClick={() => setView('video')}>Video</button>
            <button role="tab" aria-selected={view === 'audio'} onClick={() => setView('audio')}>Audio</button>
            <button role="tab" aria-selected={view === 'three'} onClick={() => setView('three')}>3D</button>
          </div>}
        </div>
        <div className="workspace-content">
          {view === 'graph' && <GraphWorkspace />}
          <Suspense fallback={<div className="view-loading" role="status">Loading view...</div>}>
            {view === 'video' && <WebRtcVideo />}
            {view === 'audio' && <WebRtcAudio />}
            {view === 'three' && <SkeletonViewport />}
            {view === 'assets' && <AssetsWorkspace />}
            {view === 'code' && <CodeWorkspace />}
            {view === 'outputs' && <PresentationWorkspace />}
            {view === 'local' && <LocalWorkspace />}
          </Suspense>
          {connection.kind === 'offline' && <div className="connection-error">{connection.message}</div>}
        </div>
      </section>
    </main></PresentationProvider>
  );
}
