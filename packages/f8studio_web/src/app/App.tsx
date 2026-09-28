import { Activity, Archive, Boxes, CircleDot, Plug, ScrollText, Settings2, type LucideIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useEffect, useState } from 'react';

import { fetchHealth } from '../api/client';
import type { HealthStatus } from '../api/contracts';
import { GraphWorkspace } from '../graph/GraphWorkspace';
import { LogsWorkspace } from '../logs/LogsWorkspace';
import { PresentationProvider } from '../presentation/PresentationStore';
import { GraphLogDock } from './GraphLogDock';

const AssetsWorkspace = lazy(() => import('../assets/AssetsWorkspace').then((module) => ({ default: module.AssetsWorkspace })));
const CodeStateWorkspace = lazy(() => import('../editor/CodeStateWorkspace').then((module) => ({ default: module.CodeStateWorkspace })));
const PresentationWorkspace = lazy(() => import('../presentation/PresentationWorkspace').then((module) => ({ default: module.PresentationWorkspace })));
const LocalWorkspace = lazy(() => import('../local/LocalWorkspace').then((module) => ({ default: module.LocalWorkspace })));
const AgentWorkspace = lazy(() => import('../agents/AgentWorkspace').then((module) => ({ default: module.AgentWorkspace })));

type WorkspaceView = 'graph' | 'agent' | 'assets' | 'code-state' | 'outputs' | 'local' | 'logs';
interface LocationView { readonly view: WorkspaceView; readonly nodeId: string | null; readonly projectId: string | null; readonly sessionId: string | null }

function readLocationView(): LocationView {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get('view');
  const projectId = params.get('project');
  const view = requested === 'code-state' || (requested === 'agent' && projectId) ||
    WORKSPACES.some((item) => item.view === requested) ? requested as WorkspaceView : 'graph';
  return { view, nodeId: view === 'outputs' ? params.get('node') : null,
    projectId: view === 'agent' ? projectId : null, sessionId: view === 'agent' ? params.get('session') : null };
}

interface WorkspaceDefinition {
  readonly view: WorkspaceView;
  readonly label: string;
  readonly title: string;
  readonly icon: LucideIcon;
}

const WORKSPACES: readonly WorkspaceDefinition[] = [
  { view: 'graph', label: 'Graph', title: 'Graph Editor', icon: Boxes },
  { view: 'assets', label: 'Assets', title: 'Assets', icon: Archive },
  { view: 'outputs', label: 'Outputs', title: 'Live Outputs', icon: Activity },
  { view: 'local', label: 'Local integrations', title: 'Local Integrations', icon: Plug },
  { view: 'logs', label: 'Logs', title: 'Log Center', icon: ScrollText },
];

type ConnectionState =
  | { readonly kind: 'connecting' }
  | { readonly kind: 'online'; readonly health: HealthStatus }
  | { readonly kind: 'offline'; readonly message: string };

export function App() {
  const [connection, setConnection] = useState<ConnectionState>({ kind: 'connecting' });
  const [locationView, setLocationView] = useState<LocationView>(readLocationView);
  const { view, nodeId, projectId, sessionId } = locationView;
  const navigate = useCallback((nextView: WorkspaceView, nextNodeId: string | null = null) => {
    const url = new URL(window.location.href);
    url.searchParams.set('view', nextView);
    if (nextNodeId === null) url.searchParams.delete('node');
    else url.searchParams.set('node', nextNodeId);
    window.history.pushState(null, '', url);
    setLocationView({ view: nextView, nodeId: nextNodeId, projectId: null, sessionId: null });
  }, []);
  const showOutput = useCallback((outputNodeId: string) => {
    navigate('outputs', outputNodeId);
  }, [navigate]);

  useEffect(() => {
    const onPopState = () => setLocationView(readLocationView());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

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
    if (view === 'code-state' || view === 'agent') return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
      const index = Number(event.key) - 1;
      const target = WORKSPACES[index]?.view;
      if (target === undefined) return;
      event.preventDefault();
      navigate(target);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [navigate, view]);

  const statusText =
    connection.kind === 'connecting'
      ? 'Connecting'
      : connection.kind === 'online'
        ? `Local server ${connection.health.version}`
        : 'Server unavailable';

  return (
    <PresentationProvider><main className={`studio-shell${view === 'code-state' || view === 'agent' ? ' studio-shell-code-state' : ''}`}>
      <header className="topbar">
        <div className="topbar-identity">
          <div className="brand">Feel8 Studio</div>
          <h1 id="workspace-title">{view === 'code-state' ? 'Code Editor' : view === 'agent' ? 'Agent' : WORKSPACES.find((workspace) => workspace.view === view)?.title}</h1>
        </div>
        <div className={`connection connection-${connection.kind}`} role="status">
          <CircleDot size={14} aria-hidden="true" />
          <span>{statusText}</span>
        </div>
        <button className="icon-button" type="button" aria-label="Settings" title="Settings" disabled>
          <Settings2 size={18} />
        </button>
      </header>

      {view !== 'code-state' && view !== 'agent' && <aside className="rail" aria-label="Workspace navigation">
        {WORKSPACES.map(({ view: target, label, icon: Icon }) => <button
          className={`rail-button ${view === target ? 'rail-button-active' : ''}`}
          type="button"
          aria-label={label}
          title={label}
          key={target}
          onClick={() => navigate(target)}
        ><Icon size={20} /></button>)}
      </aside>}

      <section className="workspace" aria-labelledby="workspace-title">
        <div className="workspace-content">
          {view === 'graph' && <GraphLogDock onOpenLogs={() => navigate('logs')}><GraphWorkspace onShowOutput={showOutput} /></GraphLogDock>}
          <Suspense fallback={<div className="view-loading" role="status">Loading view...</div>}>
            {view === 'assets' && <AssetsWorkspace />}
            {view === 'code-state' && <CodeStateWorkspace />}
            {view === 'outputs' && <PresentationWorkspace nodeId={nodeId} />}
            {view === 'local' && <LocalWorkspace />}
            {view === 'logs' && <LogsWorkspace />}
            {view === 'agent' && projectId !== null && <AgentWorkspace projectId={projectId} initialSessionId={sessionId} />}
          </Suspense>
          {connection.kind === 'offline' && <div className="connection-error">{connection.message}</div>}
        </div>
      </section>
    </main></PresentationProvider>
  );
}
