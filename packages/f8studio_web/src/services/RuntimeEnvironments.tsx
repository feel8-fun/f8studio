import { Folder, Layers, Plus, RefreshCw, Trash2, X } from 'lucide-react';
import { useEffect, useState, type FormEvent } from 'react';

import { cancelEnvironmentPreparation, createEnvironment, fetchEnvironmentDetail, fetchRuntimeStorage,
  prepareEnvironment, removeEnvironment, retainEnvironment, setRuntimeStorage } from '../api/client';
import type { EnvironmentDetail, EnvironmentStatus, ExtensionStatus, RuntimeStorageStatus } from '../api/contracts.gen';

const bytes = (value: number) => `${(value / 1024 / 1024).toFixed(1)} MiB`;
const lines = (value: string) => value.split('\n').map((line) => line.trim()).filter(Boolean);

export function RuntimeEnvironments({ environments, extensions, onRefresh, locked }: {
  environments: readonly EnvironmentStatus[]; extensions: readonly ExtensionStatus[];
  onRefresh: () => Promise<void>; locked: boolean;
}) {
  const [name, setName] = useState('');
  const [base, setBase] = useState('');
  const [policy, setPolicy] = useState<'preserve' | 'adjust'>('preserve');
  const [python, setPython] = useState('3.12.*');
  const [conda, setConda] = useState('');
  const [pypi, setPypi] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState('');
  const [detail, setDetail] = useState<EnvironmentDetail | null>(null);
  const [storage, setStorage] = useState<RuntimeStorageStatus | null>(null);
  const [storagePath, setStoragePath] = useState('');
  const [revision, setRevision] = useState(0);
  const preparing = environments.some((environment) => environment.state === 'preparing');
  const disabled = locked || busy || preparing;
  useEffect(() => {
    const controller = new AbortController();
    void fetchRuntimeStorage(controller.signal).then((next) => {
      if (!controller.signal.aborted) { setStorage(next); setStoragePath(next.path); }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Unable to load runtime storage');
    });
    return () => controller.abort();
  }, [revision]);
  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    const controller = new AbortController();
    setDetail(null);
    void fetchEnvironmentDetail(selected, controller.signal).then((next) => {
      if (!controller.signal.aborted) setDetail(next);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Unable to inspect environment');
    });
    return () => controller.abort();
  }, [selected, revision, preparing]);
  const act = async (action: () => Promise<unknown>) => {
    setBusy(true); setError('');
    try { await action(); await onRefresh(); setRevision((value) => value + 1); }
    catch (reason: unknown) { setError(reason instanceof Error ? reason.message : 'Environment operation failed'); }
    finally { setBusy(false); }
  };
  const create = (event: FormEvent) => {
    event.preventDefault();
    void act(async () => {
      const next = await createEnvironment({ name, baseEnvironmentId: base || null, policy, python,
        condaDependencies: lines(conda), pypiDependencies: lines(pypi) });
      setSelected(next.environmentId); setName('');
    });
  };
  return <section className="runtime-environments" aria-label="Runtime environments">
    <header><div><h2><Layers size={18} />Runtime environments</h2><p>Inspect dependencies, prepare runtimes, and create independent developer revisions.</p></div></header>
    {error && <div role="alert" className="services-error">{error}</div>}
    <div className="runtime-grid">
      {environments.map((environment) => {
        const state = environment.state ?? (environment.ready ? 'ready' : 'missing');
        const users = environment.extensionIds.map((id) => extensions.find((extension) => extension.extensionId === id)?.name ?? id);
        return <article className="runtime-card" key={environment.environmentId}>
          <div className="extension-heading"><strong><button type="button" className="runtime-name" onClick={() => { setError(''); setSelected(environment.environmentId); }}>{environment.name || environment.environmentId}</button></strong><span className={`extension-state runtime-${state}`}>{state === 'changed' ? 'Definition changed' : state}</span></div>
          <div className="extension-detail">{environment.source ?? 'official'} · {environment.runtimeKind} · {environment.revision || 'unversioned'}</div>
          {environment.baseEnvironmentId && <div className="extension-detail">Based on {environments.find((item) => item.environmentId === environment.baseEnvironmentId)?.name ?? environment.baseEnvironmentId}</div>}
          <p className="runtime-users">{users.length ? users.join(', ') : 'No extension references this environment.'}</p>
          {environment.detail && <p className="extension-detail" role="status">{environment.detail}</p>}
          {!!environment.serviceClasses?.length && <details><summary>Services ({environment.serviceClasses.length})</summary><ul>{environment.serviceClasses.map((service) => <li key={service}><code>{service}</code></li>)}</ul></details>}
          {!!environment.toolIds?.length && <details><summary>Tools ({environment.toolIds.length})</summary><ul>{environment.toolIds.map((tool) => <li key={tool}><code>{tool}</code></li>)}</ul></details>}
          <div className="runtime-actions">
            {state === 'preparing' ? <button className="command-button" disabled={busy} onClick={() => void act(() => cancelEnvironmentPreparation(environment.environmentId))}><X size={14} />Cancel preparation</button>
              : <button className="command-button" disabled={disabled || environment.runtimeKind === 'bundled'} onClick={() => void act(() => prepareEnvironment(environment.environmentId))}><RefreshCw size={14} />{state === 'changed' ? 'Verify and update' : state === 'ready' ? 'Verify' : 'Prepare'}</button>}
            {environment.source === 'developer' && <>
              <label className="extension-toggle"><input type="checkbox" aria-label={`Keep ${environment.name}`} checked={environment.pinned ?? false} disabled={disabled} onChange={(event) => void act(() => retainEnvironment(environment.environmentId, event.target.checked))} />Keep revision</label>
              <button className="command-button" aria-label={`Remove ${environment.name}`} disabled={disabled || environment.pinned || environment.extensionIds.length > 0} onClick={() => void act(async () => { await removeEnvironment(environment.environmentId); if (selected === environment.environmentId) setSelected(''); })}><Trash2 size={14} />Remove unused</button>
            </>}
          </div>
        </article>;
      })}
    </div>
    {selected && <article className="extension-content runtime-inspector" aria-label="Environment details">
      <div className="extension-heading"><h2>{detail?.name ?? 'Loading environment…'}</h2><button className="icon-button" aria-label="Close environment details" onClick={() => setSelected('')}><X size={16} /></button></div>
      {detail && <>
        <dl className="extension-metadata"><dt>Revision</dt><dd>{detail.revision}</dd><dt>Policy</dt><dd>{detail.policy === 'preserve' ? 'Preserve base package versions' : detail.policy === 'adjust' ? 'Allow dependency changes' : 'Publisher definition'}</dd><dt>Environment path</dt><dd><code>{detail.storagePath}</code></dd><dt>Package cache</dt><dd><code>{detail.cachePath}</code></dd>
          <dt>Logical file size</dt><dd>{bytes(detail.usage.logicalBytes ?? 0)}</dd><dt>Unique file data size</dt><dd>{bytes(detail.usage.uniqueFileBytes ?? 0)}</dd><dt>Files with multiple hardlinks</dt><dd>{bytes(detail.usage.sharedLinkBytes ?? 0)}</dd><dt>Files with a single hardlink</dt><dd>{bytes(detail.usage.exclusiveFileBytes ?? 0)}</dd>
        </dl>
        <p className="extension-muted">File sizes are logical bytes, not physical disk allocation. Hardlinks share data; copy-on-write savings are not measured here. Removing a revision keeps the shared package cache.</p>
        {!!detail.changedPackages?.length && <details><summary>Changes from base ({detail.changedPackages.length})</summary><ul>{detail.changedPackages.map((change) => <li key={change}><code>{change}</code></li>)}</ul></details>}
        <details><summary>Environment definition</summary><pre>{detail.manifest}</pre></details>
      </>}
    </article>}
    <details className="runtime-create"><summary><Plus size={16} />Create developer environment</summary>
      <form className="runtime-form" onSubmit={create}>
        <label>Name<input aria-label="Environment name" required pattern="[a-z][a-z0-9_-]{0,63}" value={name} onChange={(event) => setName(event.target.value)} placeholder="my-game-tools" /></label>
        <label>Base environment<select aria-label="Base environment" value={base} onChange={(event) => setBase(event.target.value)}><option value="">Independent environment</option>{environments.filter((environment) => environment.runtimeKind !== 'bundled' && environment.state !== 'preparing' && (environment.source !== 'developer' || environment.ready)).map((environment) => <option key={environment.environmentId} value={environment.environmentId}>{environment.name || environment.environmentId} · {environment.source} · {environment.revision}</option>)}</select></label>
        {base ? <label>Dependency policy<select aria-label="Dependency policy" value={policy} onChange={(event) => setPolicy(event.target.value === 'adjust' ? 'adjust' : 'preserve')}><option value="preserve">Preserve base package versions</option><option value="adjust">Allow changes to base dependencies</option></select></label>
          : <label>Python version<input aria-label="Python version" required value={python} onChange={(event) => setPython(event.target.value)} /></label>}
        <label>Conda requirements<textarea aria-label="Conda requirements" value={conda} onChange={(event) => setConda(event.target.value)} placeholder={'numpy >=1.26,<3\nffmpeg >=7'} /></label>
        <label>PyPI requirements<textarea aria-label="PyPI requirements" value={pypi} onChange={(event) => setPypi(event.target.value)} placeholder={'requests>=2.32,<3\nPillow>=11'} /></label>
        <p>One requirement per line. Creating saves a snapshot only; Prepare resolves and installs it independently. Incompatible requirements report an error without modifying the base.</p>
        <button className="command-button primary" disabled={disabled || !name}><Plus size={14} />Create revision</button>
      </form>
    </details>
    <details className="runtime-storage"><summary><Folder size={16} />Runtime storage and shared cache</summary>
      {storage && <form className="runtime-form" onSubmit={(event) => { event.preventDefault(); void act(() => setRuntimeStorage(storagePath)); }}>
        <label>Storage directory<input aria-label="Runtime storage directory" value={storagePath} disabled={!storage.canChange || disabled} onChange={(event) => setStoragePath(event.target.value)} /></label>
        <p>Managed environments and their package cache use this directory on the same volume. Workspace environments remain in their development checkout. Models and resources remain separate.</p>
        {!storage.canChange && <p>Remove prepared managed environments before changing storage. Existing directories are not moved.</p>}
        <button className="command-button" disabled={!storage.canChange || disabled || !storagePath}>Save storage directory</button>
      </form>}
    </details>
  </section>;
}
