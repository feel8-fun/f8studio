import { ArrowLeft, ArrowRight, Download, PackagePlus, RefreshCw, Search, Trash2, X, XCircle } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { cancelExtensionInstall, fetchEnvironments, fetchExtensions,
  importExtensionPackage, installExtension, setExtensionEnabled, uninstallExtension } from '../api/client';
import type { EnvironmentStatus, ExtensionStatus } from '../api/contracts';

import { ExtensionDetails, ExtensionLink, extensionHref, readExtensionLocation, type ExtensionLocation } from './ExtensionDetails';

export function ServicesWorkspace() {
  const [extensions, setExtensions] = useState<readonly ExtensionStatus[]>([]);
  const [environments, setEnvironments] = useState<readonly EnvironmentStatus[]>([]);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [detailRevision, setDetailRevision] = useState(0);
  const [location, setLocation] = useState(readExtensionLocation);
  const [loaded, setLoaded] = useState(false);
  const navigate = useCallback((next: ExtensionLocation | null) => {
    window.history.pushState(null, '', extensionHref(next));
    setLocation(next);
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
  }, []);
  useEffect(() => {
    const onPopState = () => setLocation(readExtensionLocation());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [packageUrl, setPackageUrl] = useState('');
  const [packageHash, setPackageHash] = useState('');

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [nextExtensions, nextEnvironments] = await Promise.all([
        fetchExtensions(signal), fetchEnvironments(signal),
      ]);
      if (signal?.aborted) return;
      setExtensions(nextExtensions);
      setLoaded(true);
      setEnvironments(nextEnvironments);
      setError('');
    } catch (reason: unknown) {
      if (!signal?.aborted) setError(reason instanceof Error ? reason.message : 'Unable to load extensions');
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const installing = extensions.some((extension) => extension.state === 'installing');
  useEffect(() => {
    if (!installing) return;
    const controller = new AbortController();
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      try {
        const statuses = await fetchExtensions(controller.signal);
        if (controller.signal.aborted) return;
        if (statuses.some((extension) => extension.state === 'installing')) setExtensions(statuses);
        else await load(controller.signal);
      } catch (reason: unknown) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Unable to check installation');
      } finally {
        pending = false;
      }
    };
    const timer = window.setInterval(() => void poll(), 1000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [installing, load]);

  const act = useCallback(async (action: () => Promise<ExtensionStatus>, refresh: boolean, rollback?: ExtensionStatus) => {
    setBusy(true);
    setError('');
    try {
      const status = await action();
      setExtensions((current) => current.map((extension) => extension.extensionId === status.extensionId ? status : extension));
      if (refresh) await load();
    } catch (reason: unknown) {
      if (rollback) setExtensions((current) => current.map((extension) => extension.extensionId === rollback.extensionId ? rollback : extension));
      setError(reason instanceof Error ? reason.message : 'Extension operation failed');
    } finally {
      setBusy(false);
    }
  }, [load]);

  const toggle = useCallback((extension: ExtensionStatus, enabled: boolean) => {
    const updated: ExtensionStatus = { ...extension, state: enabled ? 'installed' : 'disabled' };
    setExtensions((current) => current.map((item) => item.extensionId === extension.extensionId ? updated : item));
    return act(() => setExtensionEnabled(extension.extensionId, enabled), true, extension);
  }, [act]);

  const filteredExtensions = useMemo(() => extensions.filter((extension) =>
    `${extension.name} ${extension.extensionId} ${extension.description}`.toLowerCase().includes(query.trim().toLowerCase())), [extensions, query]);
  const locked = busy || installing;

  const importPackage = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      setExtensions(await importExtensionPackage(packageUrl.trim(), packageHash.trim().toLowerCase()));
      setPackageUrl('');
      setPackageHash('');
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : 'Unable to import extension package');
    } finally {
      setBusy(false);
    }
  }, [packageUrl, packageHash]);

  const selectedExtension = extensions.find((extension) => extension.extensionId === location?.extensionId);
  const extensionCard = (extension: ExtensionStatus, showDetailsLink: boolean) => (
    <div className="extension-row" key={extension.extensionId}>
      <div className="extension-heading"><PackagePlus size={17} /><strong>{showDetailsLink ? <ExtensionLink location={{ extensionId: extension.extensionId }} onNavigate={navigate}>{extension.name}</ExtensionLink> : extension.name}</strong><span className={`extension-state extension-${extension.state}`}>{extension.state}</span></div>
      <div className="extension-classes">v{extension.version} · {extension.serviceClasses.length} services · {extension.toolIds?.length ?? 0} tools · {extension.skillIds?.length ?? 0} skills</div>
      <div className="extension-detail">{extension.description}</div>
      {extension.preinstalled && <div className="extension-detail">Included with this distribution.</div>}
      {extension.runtimeKind === 'shared' && <div className="extension-detail">Reuses an installed official environment. No additional environment download.</div>}
      {(extension.state === 'available' || extension.state === 'failed') && extension.runtimeKind === 'pixi' && <div className="extension-detail">Runtime dependencies may need to be downloaded. Shared runtimes are reused.</div>}
      {extension.detail && <div className={extension.state === 'failed' ? 'extension-error' : 'extension-detail'} role="status">{extension.state === 'failed' && <XCircle size={14} />}{extension.detail}</div>}
      {(extension.state === 'available' || extension.state === 'failed') && <button className="command-button primary" type="button" disabled={locked} aria-label={`Install ${extension.name}`} onClick={() => void act(() => installExtension(extension.extensionId), false)}><Download size={15} />{extension.state === 'failed' ? 'Retry install' : 'Install'}</button>}
      {extension.state === 'installing' && <button className="command-button" type="button" disabled={busy} aria-label={`Cancel ${extension.name} installation`} onClick={() => void act(() => cancelExtensionInstall(extension.extensionId), true)}><X size={15} />Cancel</button>}
      {(extension.state === 'installed' || extension.state === 'disabled') && <div className="extension-actions">
        <label className="extension-toggle"><input type="checkbox" aria-label={`Enable ${extension.name}`} checked={extension.state === 'installed'} disabled={locked} onChange={(event) => void toggle(extension, event.target.checked)} />Enabled</label>
        <button className="command-button" type="button" disabled={locked} aria-label={`Uninstall ${extension.name}`} onClick={() => void act(() => uninstallExtension(extension.extensionId), true)}><Trash2 size={15} />Uninstall</button>
      </div>}
      {showDetailsLink && <ExtensionLink className="extension-details-link" location={{ extensionId: extension.extensionId }} onNavigate={navigate}>View details<ArrowRight size={14} /></ExtensionLink>}
    </div>
  );

  return <div className="services-workspace">
    <div className="services-toolbar">
      {location ? <ExtensionLink className="extension-back" location={null} onNavigate={navigate}><ArrowLeft size={15} />All extensions</ExtensionLink> : <label className="services-search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search extensions" aria-label="Search extensions" /></label>}
      <span>{extensions.length} extensions</span>
      <button className="icon-button" type="button" title="Refresh extensions" aria-label="Refresh extensions" onClick={() => { setDetailRevision((value) => value + 1); void load(); }}><RefreshCw size={16} /></button>
    </div>
    {error && <div className="services-error" role="alert">{error}</div>}
    <div className="services-body" ref={bodyRef}>
      {location ? <section className="services-extensions extension-detail-page" aria-label="Extension details">
        {selectedExtension ? <>
          {extensionCard(selectedExtension, false)}
          <p className="extension-lifecycle-note">Installation and enabling apply to the entire extension, including its services, tools and skills.</p>
          <ExtensionDetails key={`${selectedExtension.extensionId}/${selectedExtension.version}`} location={location} onNavigate={navigate} refreshRevision={detailRevision} />
        </> : <div className="services-empty">{loaded ? 'Extension not found.' : 'Loading extension…'}</div>}
      </section> : <section className="services-extensions" aria-label="Extensions">
        <header><h2>Extensions</h2></header>
        <div className="extension-grid">
        {filteredExtensions.map((extension) => extensionCard(extension, true))}
        </div>
        {extensions.length > 0 && filteredExtensions.length === 0 && <div className="services-empty">No matching extensions</div>}
        {extensions.length === 0 && <div className="services-empty">No extension catalog is available in this build.</div>}
        <form className="extension-import" onSubmit={(event) => { event.preventDefault(); void importPackage(); }}>
          <h3>Add extension package</h3>
          <p>Use the HTTPS package link and SHA-256 provided by a trusted extension publisher.</p>
          <label>Package URL<input aria-label="Extension package URL" type="url" required value={packageUrl} onChange={(event) => setPackageUrl(event.target.value)} placeholder="https://…/extension.zip" /></label>
          <label>SHA-256<input aria-label="Extension package SHA-256" required pattern="[a-fA-F0-9]{64}" value={packageHash} onChange={(event) => setPackageHash(event.target.value)} /></label>
          <button type="submit" className="command-button" disabled={locked || !packageUrl || !/^[a-fA-F0-9]{64}$/.test(packageHash.trim())}>Add package</button>
        </form>
        {environments.length > 0 && <div className="extension-environments" aria-label="Shared runtimes">
          <h3>Shared runtimes</h3>
          {environments.map((environment) => <div className="extension-detail" key={environment.environmentId}>
            {environment.extensionIds.map((id) => extensions.find((extension) => extension.extensionId === id)?.name ?? id).join(', ')}
            {' · '}{environment.ready ? 'ready' : 'not ready'}
          </div>)}
        </div>}
      </section>}
    </div>
  </div>;
}
