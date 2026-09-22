import { AudioLines, Boxes, CircleDot, Cuboid, Settings2, Video } from 'lucide-react';
import { useEffect, useState } from 'react';

import { fetchHealth } from '../api/client';
import type { HealthStatus } from '../api/contracts';
import { WebRtcVideo } from '../media/WebRtcVideo';
import { WebRtcAudio } from '../media/WebRtcAudio';
import { SkeletonViewport } from '../three/SkeletonViewport';

type ConnectionState =
  | { readonly kind: 'connecting' }
  | { readonly kind: 'online'; readonly health: HealthStatus }
  | { readonly kind: 'offline'; readonly message: string };

export function App() {
  const [connection, setConnection] = useState<ConnectionState>({ kind: 'connecting' });
  const [view, setView] = useState<'video' | 'audio' | 'three'>('video');

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

  const statusText =
    connection.kind === 'connecting'
      ? 'Connecting'
      : connection.kind === 'online'
        ? `Local server ${connection.health.version}`
        : 'Server unavailable';

  return (
    <main className="studio-shell">
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
        <button className={`rail-button ${view === 'video' ? 'rail-button-active' : ''}`} type="button" aria-label="Video" title="Video" onClick={() => setView('video')}>
          <Video size={20} />
        </button>
        <button className={`rail-button ${view === 'three' ? 'rail-button-active' : ''}`} type="button" aria-label="3D" title="3D" onClick={() => setView('three')}>
          <Cuboid size={20} />
        </button>
        <button className={`rail-button ${view === 'audio' ? 'rail-button-active' : ''}`} type="button" aria-label="Audio" title="Audio" onClick={() => setView('audio')}>
          <AudioLines size={20} />
        </button>
        <button className="rail-button" type="button" aria-label="Graph" title="Graph" disabled><Boxes size={20} /></button>
      </aside>

      <section className="workspace" aria-labelledby="workspace-title">
        <div className="workspace-toolbar">
          <h1 id="workspace-title">Media Lab</h1>
          <div className="view-tabs" role="tablist" aria-label="Media view">
            <button role="tab" aria-selected={view === 'video'} onClick={() => setView('video')}>Video</button>
            <button role="tab" aria-selected={view === 'audio'} onClick={() => setView('audio')}>Audio</button>
            <button role="tab" aria-selected={view === 'three'} onClick={() => setView('three')}>3D</button>
          </div>
        </div>
        <div className="workspace-content">
          {view === 'video' && <WebRtcVideo />}
          {view === 'audio' && <WebRtcAudio />}
          {view === 'three' && <SkeletonViewport />}
          {connection.kind === 'offline' && <div className="connection-error">{connection.message}</div>}
        </div>
      </section>
    </main>
  );
}
