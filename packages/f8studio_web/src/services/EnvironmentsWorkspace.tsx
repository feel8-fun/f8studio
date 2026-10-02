import { RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import { fetchEnvironments, fetchExtensions } from '../api/client';
import type { EnvironmentStatus, ExtensionStatus } from '../api/contracts';
import { RuntimeEnvironments } from './RuntimeEnvironments';

export function EnvironmentsWorkspace() {
  const [environments, setEnvironments] = useState<readonly EnvironmentStatus[]>([]);
  const [extensions, setExtensions] = useState<readonly ExtensionStatus[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [nextEnvironments, nextExtensions] = await Promise.all([
        fetchEnvironments(signal), fetchExtensions(signal),
      ]);
      if (signal?.aborted) return;
      setEnvironments(nextEnvironments);
      setExtensions(nextExtensions);
      setLoaded(true);
      setError('');
    } catch (reason: unknown) {
      if (!signal?.aborted) setError(reason instanceof Error ? reason.message : 'Unable to load runtime environments');
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const working = environments.some((environment) => environment.state === 'preparing') ||
    extensions.some((extension) => extension.state === 'installing');
  useEffect(() => {
    if (!working) return;
    const controller = new AbortController();
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      try { await load(controller.signal); }
      finally { pending = false; }
    };
    const timer = window.setInterval(() => void poll(), 1000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [working, load]);

  return <div className="services-workspace environments-workspace">
    <div className="services-toolbar">
      <span>{loaded ? `${environments.length} environments` : 'Loading environments…'}</span>
      <button className="icon-button" type="button" title="Refresh runtime environments" aria-label="Refresh runtime environments" onClick={() => void load()}><RefreshCw size={16} /></button>
    </div>
    {error && <div className="services-error" role="alert">{error}</div>}
    <div className="services-body">
      <div className="services-extensions">
        {loaded && <RuntimeEnvironments environments={environments} extensions={extensions} onRefresh={load} locked={working} />}
      </div>
    </div>
  </div>;
}
