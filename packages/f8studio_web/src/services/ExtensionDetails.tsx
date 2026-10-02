import { ArrowLeft, ArrowRight, FileText, Layers, Wrench } from 'lucide-react';
import { useEffect, useState, type MouseEvent, type ReactNode } from 'react';

import { fetchExtensionDetail } from '../api/client';
import type { ExtensionDetail, F8ServiceDescribe } from '../api/contracts.gen';

export type ExtensionLocation = { extensionId: string; kind?: 'service' | 'tool' | 'skill'; itemId?: string };

export function readExtensionLocation(): ExtensionLocation | null {
  const params = new URLSearchParams(window.location.search);
  const extensionId = params.get('extension');
  if (!extensionId) return null;
  for (const kind of ['service', 'tool', 'skill'] as const) {
    const itemId = params.get(kind);
    if (itemId) return { extensionId, kind, itemId };
  }
  return { extensionId };
}

export function extensionHref(location: ExtensionLocation | null): string {
  const params = new URLSearchParams({ view: 'extensions' });
  if (location) {
    params.set('extension', location.extensionId);
    if (location.kind && location.itemId) params.set(location.kind, location.itemId);
  }
  return `?${params}`;
}

export function ExtensionLink({ location, onNavigate, children, className }: {
  location: ExtensionLocation | null; onNavigate: (location: ExtensionLocation | null) => void;
  children: ReactNode; className?: string;
}) {
  const follow = (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); onNavigate(location);
  };
  return <a href={extensionHref(location)} onClick={follow} className={className}>{children}</a>;
}

function ServiceDetails({ describe }: { describe: F8ServiceDescribe }) {
  const service = describe.service;
  const sections = [
    { title: 'Input ports', items: service.dataInPorts ?? [] },
    { title: 'Output ports', items: service.dataOutPorts ?? [] },
    { title: 'State fields', items: service.stateFields ?? [] },
    { title: 'Commands', items: service.commands ?? [] },
  ];
  return <>
    <h2>{service.label}</h2><code>{service.serviceClass}</code>
    <p>{service.description || 'No description provided.'}</p>
    {sections.map(({ title, items }) => <section className="extension-spec-section" key={title}>
      <h3>{title} <span>{items.length}</span></h3>
      {items.length ? items.map((item) => <details key={item.name} className="extension-spec-item">
        <summary><strong>{item.name}</strong><span>{item.description}</span></summary>
        <pre>{JSON.stringify(item, null, 2)}</pre>
      </details>) : <p className="extension-muted">None declared.</p>}
    </section>)}
    <section className="extension-spec-section"><h3>Operators <span>{describe.operators?.length ?? 0}</span></h3>
      {(describe.operators ?? []).map((operator) => <details className="extension-spec-item" key={operator.operatorClass}>
        <summary><strong>{operator.label}</strong><code>{operator.operatorClass}</code></summary>
        <p>{operator.description}</p><pre>{JSON.stringify(operator, null, 2)}</pre>
      </details>)}
      {!describe.operators?.length && <p className="extension-muted">None declared.</p>}
    </section>
  </>;
}

export function ExtensionDetails({ location, onNavigate, refreshRevision }: {
  location: ExtensionLocation; onNavigate: (location: ExtensionLocation | null) => void; refreshRevision: number;
}) {
  const [detail, setDetail] = useState<ExtensionDetail | null>(null);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setDetail(null); setError('');
    void fetchExtensionDetail(location.extensionId, controller.signal).then((next) => {
      if (!controller.signal.aborted) setDetail(next);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Unable to load extension details');
    });
    return () => controller.abort();
  }, [location.extensionId, revision, refreshRevision]);
  if (error) return <div className="extension-content"><div role="alert" className="extension-error">{error}</div><button className="command-button" onClick={() => setRevision((value) => value + 1)}>Retry details</button></div>;
  if (!detail) return <div className="extension-content" role="status">Loading extension details…</div>;
  const base = { extensionId: location.extensionId };
  const service = detail.services.find((item) => item.serviceClass === location.itemId);
  const tool = detail.tools.find((item) => item.toolId === location.itemId);
  const skill = detail.skills.find((item) => item.skillId === location.itemId);
  if (location.kind) return <article className="extension-content" aria-label={`${location.kind} details`}>
    <ExtensionLink className="extension-back" location={base} onNavigate={onNavigate}><ArrowLeft size={15} />Back to extension contents</ExtensionLink>
    {location.kind === 'service' && service && (service.describe ? <ServiceDetails describe={service.describe} /> : <>
      <h2>{service.serviceClass}</h2><p>The publisher has not included a service description in this package.</p>
    </>)}
    {location.kind === 'tool' && tool && <>
      <h2>{tool.name}</h2><code>{tool.toolId}</code><p>{tool.description}</p>
      <dl className="extension-metadata">
        <dt>Confirmation</dt><dd>{tool.requiresConfirmation ? 'Required before running' : 'Not required'}</dd>
        <dt>Run duration</dt><dd>{tool.timeoutSeconds === null ? 'Runs until stopped' : `Up to ${tool.timeoutSeconds ?? 300} seconds`}</dd>
        <dt>Concurrent tasks</dt><dd>{tool.allowConcurrent ? 'Supported' : 'Not supported'}</dd>
        <dt>Platforms</dt><dd>{tool.platforms?.join(', ') || 'All platforms'}</dd>
      </dl>
      <section className="extension-spec-section"><h3>Parameters <span>{tool.fields?.length ?? 0}</span></h3>
        {(tool.fields ?? []).map((field) => <div className="extension-spec-item" key={field.name}>
          <strong>{field.label}</strong><code>{field.name}</code>
          <p>{field.kind ?? 'string'} · {field.required ? 'Required' : 'Optional'}{field.default != null && ` · Default: ${String(field.default)}`}</p>
          {!!field.choices?.length && <p>Choices: {field.choices.join(', ')}</p>}
        </div>)}
        {!tool.fields?.length && <p className="extension-muted">No parameters.</p>}
      </section>
    </>}
    {location.kind === 'skill' && skill && <><h2>{skill.skillId}</h2><p>Agent instructions supplied by this extension.</p><pre className="extension-skill-content">{skill.content}</pre></>}
    {((location.kind === 'service' && !service) || (location.kind === 'tool' && !tool) || (location.kind === 'skill' && !skill)) && <p role="alert">This item is not declared by the extension.</p>}
  </article>;
  const groups = [
    { title: 'Services', icon: Layers, kind: 'service' as const, items: detail.services.map((item) => ({ id: item.serviceClass, name: item.describe?.service.label ?? item.serviceClass, description: item.describe?.service.description ?? '' })) },
    { title: 'Tools', icon: Wrench, kind: 'tool' as const, items: detail.tools.map((item) => ({ id: item.toolId, name: item.name, description: item.description })) },
    { title: 'Skills', icon: FileText, kind: 'skill' as const, items: detail.skills.map((item) => ({ id: item.skillId, name: item.skillId, description: 'View agent instructions' })) },
  ];
  return <div className="extension-capability-groups">
    {groups.map(({ title, icon: Icon, kind, items }) => <section className="extension-content" key={kind} aria-label={title}>
      <h2><Icon size={18} />{title}<span>{items.length}</span></h2>
      {items.map((item) => <ExtensionLink className="extension-capability-link" key={item.id} location={{ ...base, kind, itemId: item.id }} onNavigate={onNavigate}>
        <div><strong>{item.name}</strong><code>{item.id}</code>{item.description && <p>{item.description}</p>}</div><ArrowRight size={16} />
      </ExtensionLink>)}
      {!items.length && <p className="extension-muted">No {kind === 'service' ? 'services' : kind === 'tool' ? 'tools' : 'skills'} declared.</p>}
    </section>)}
  </div>;
}
