import { Check, Plus, Search, Settings2, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { createAgentConnection, deleteAgentConnection, fetchAgentProviderSettings, saveAgentProviderSettings } from '../api/client';
import type { AgentProviderSettings, CreateAgentConnection } from '../api/contracts';
import { AgentModelPicker } from './AgentModelPicker';
import { ModelCapabilitiesEditor } from './ModelCapabilitiesEditor';
import { setCapability } from './modelCapabilities';
import { useConnectionModels } from './useConnectionModels';

export const AGENT_PROVIDERS_CHANGED = 'studio-agent-providers-changed';

type AgentProtocol = CreateAgentConnection['protocol'];
const DEFAULT_ENDPOINTS: Record<AgentProtocol, string> = {
  openai_responses: 'https://api.openai.com/v1',
  openai_chat: 'http://127.0.0.1:11434/v1',
  anthropic: 'https://api.anthropic.com/v1',
  systemone: 'https://api.typesafe.ai/v1',
};

function connectionProtocol(settings: AgentProviderSettings): AgentProtocol | null {
  if (settings.protocol) return settings.protocol;
  if (settings.providerId === 'openai') return 'openai_responses';
  if (settings.providerId === 'anthropic') return 'anthropic';
  if (settings.kind === 'decision') return 'systemone';
  if (settings.providerId === 'google_gemini' || settings.providerId === 'ollama') return 'openai_chat';
  return null;
}

function ProviderForm({ settings, onSaved, onDeleted }: {
  readonly settings: AgentProviderSettings;
  readonly onSaved: (settings: AgentProviderSettings) => void;
  readonly onDeleted: (providerId: string) => void;
}) {
  const [name, setName] = useState(settings.displayName);
  const [endpoint, setEndpoint] = useState(settings.endpoint);
  const [apiKey, setApiKey] = useState('');
  const [clearApiKey, setClearApiKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const protocol = connectionProtocol(settings);
  const { model, setModel, models, setModels, modelCapabilities, setModelCapabilities,
    probe, setProbe, test } = useConnectionModels({
      settings, protocol, endpoint: endpoint.trim() || (protocol ? DEFAULT_ENDPOINTS[protocol] : ''), apiKey, providerId: clearApiKey ? '' : settings.providerId,
      setBusy, setError,
    });

  const save = async () => {
    setBusy(true); setError(''); setSaved(false);
    try {
      const savedModels = [...new Set([...models, model.trim()])];
      const next = await saveAgentProviderSettings(settings.providerId, {
        model: model.trim(), endpoint: endpoint.trim(), apiKey: apiKey.trim() || undefined, clearApiKey,
        ...(settings.custom ? { displayName: name.trim() } : {}), models: savedModels, modelCapabilities,
        supportsImage: false,
      });
      setModel(next.model); setModels(next.models ?? (next.model ? [next.model] : []));
      setModelCapabilities(next.modelCapabilities ?? []);
      setName(next.displayName); setEndpoint(next.endpoint);
      setApiKey(''); setClearApiKey(false); setSaved(true);
      onSaved(next);
      window.dispatchEvent(new Event(AGENT_PROVIDERS_CHANGED));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };

  const remove = async () => {
    setBusy(true); setError('');
    try {
      await deleteAgentConnection(settings.providerId);
      onDeleted(settings.providerId);
      window.dispatchEvent(new Event(AGENT_PROVIDERS_CHANGED));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };

  return <form className="agent-settings-form" onSubmit={(event) => { event.preventDefault(); void save(); }} onChange={() => { setSaved(false); setProbe(null); }}>
    <div className="agent-settings-heading"><div><h3>{settings.displayName}</h3>{protocol && <span className="agent-protocol-label">{protocol.replaceAll('_', ' ')}</span>}</div><span className={settings.configured ? 'agent-connection-ready' : 'agent-connection-pending'}>{settings.configured ? 'Configured' : 'Needs setup'}</span></div>
    {!settings.custom && settings.kind === 'decision' && <p>{settings.providerId === 'typesafe' ? 'Typed decisions for graph nodes: Choice, Score, and Noul. Jev currently accepts text and JSON; image input is not supported.' : 'External System-One host for typed decisions.'}</p>}
    <fieldset disabled={busy}>
      {settings.custom && <label>Connection name<input required maxLength={80} value={name} onChange={(event) => setName(event.target.value)} /></label>}
      {settings.kind === 'decision' && <label>Decision node provider ID<input readOnly value={settings.providerId} onFocus={(event) => event.target.select()} /></label>}
      <div className="agent-settings-connection-fields">
        <label>Base URL<input type="url" value={endpoint} onChange={(event) => { setEndpoint(event.target.value); setModels([]); setModelCapabilities([]); }} required={settings.providerId === 'ollama' || settings.providerId === 'google_gemini' || settings.kind === 'decision'} placeholder="Default provider endpoint" /></label>
        {(settings.requiresApiKey || settings.providerId === 'systemone_local' || settings.custom) && <label>API key<input type="password" autoComplete="new-password" value={apiKey} disabled={clearApiKey} onChange={(event) => { setApiKey(event.target.value); setModels([]); setModelCapabilities([]); }} placeholder={settings.apiKeySet ? 'Key configured — leave blank to keep it' : 'Enter API key'} /></label>}
      </div>
      {settings.apiKeySet && (settings.requiresApiKey || settings.providerId === 'systemone_local' || settings.custom) && <label className="agent-settings-checkbox"><input type="checkbox" checked={clearApiKey} onChange={(event) => { setClearApiKey(event.target.checked); setApiKey(''); setModels([]); setModelCapabilities([]); }} />Remove saved API key</label>}
      <div className="agent-settings-model-row"><AgentModelPicker label="Default model" models={models} value={model} onChange={setModel} />
        {protocol && <div className="agent-settings-actions"><button className="icon-button" type="button" aria-label="Refresh models" title="Refresh models" onClick={() => void test(false)}><Search size={16} /></button><button className="command-button" type="button" disabled={!model.trim()} onClick={() => void test(true)}>Test model</button></div>}
      </div>
      <ModelCapabilitiesEditor model={model} capabilities={modelCapabilities} onChange={(field, value) => setModelCapabilities((current) => setCapability(current, model, field, value))} />
      {probe && <p role="status" className={probe.connected ? 'agent-probe-ok' : 'agent-error'}>{probe.detail}</p>}
      <div className="agent-settings-footer"><button className="command-button primary" type="submit" aria-label="Save provider" disabled={!model.trim()}><Check size={15} />{busy ? 'Saving…' : 'Save'}</button>
        {confirmDelete ? <div className="agent-settings-actions"><span>Existing sessions will need another connection.</span><button className="command-button" type="button" onClick={() => setConfirmDelete(false)}>Cancel</button><button className="command-button danger" type="button" onClick={() => void remove()}>Delete</button></div> : <button className="icon-button danger" type="button" aria-label="Delete connection" title="Delete connection" onClick={() => setConfirmDelete(true)}><Trash2 size={16} /></button>}
      </div>
    </fieldset>
    {saved && <p role="status">Saved. {settings.configured ? settings.kind === 'decision' ? 'Available to Decision nodes in your graph.' : 'Available for new agent sessions.' : 'Add an API key to enable this provider.'} {probe?.connected ? probe.verified === 'model' ? 'Model inference was tested.' : 'Model listing was tested; inference was not.' : 'Connection has not been tested.'}</p>}
    {error && <p className="agent-error" role="alert">{error}</p>}
  </form>;
}

function NewConnectionForm({ onCreated }: { readonly onCreated: (settings: AgentProviderSettings) => void }) {
  const [name, setName] = useState('');
  const [protocol, setProtocol] = useState<AgentProtocol>('openai_responses');
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINTS.openai_responses);
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const { model, setModel, models, setModels, modelCapabilities, setModelCapabilities,
    probe, setProbe, test } = useConnectionModels({ protocol, endpoint, apiKey, setBusy, setError });

  const create = async () => {
    setBusy(true); setError('');
    try {
      const created = await createAgentConnection({
        displayName: name.trim(), protocol, endpoint: endpoint.trim(), apiKey: apiKey.trim(),
        model: model.trim(), models: [...new Set([...models, model.trim()])], supportsImage: false, modelCapabilities,
      });
      setApiKey('');
      onCreated(created);
      window.dispatchEvent(new Event(AGENT_PROVIDERS_CHANGED));
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };

  return <form className="agent-settings-form" onChange={() => setProbe(null)} onSubmit={(event) => { event.preventDefault(); void create(); }}>
    <div className="agent-settings-heading"><h3>New connection</h3></div>
    <fieldset disabled={busy}>
      <div className="agent-settings-connection-fields"><label>Connection name<input required maxLength={80} value={name} onChange={(event) => setName(event.target.value)} placeholder="Work account, local server…" /></label>
      <label>API protocol<select aria-label="API protocol" value={protocol} onChange={(event) => {
        const next = event.target.value as AgentProtocol;
        setProtocol(next); setEndpoint(DEFAULT_ENDPOINTS[next]); setModel(''); setModels([]); setModelCapabilities([]); setProbe(null);
      }}><option value="openai_responses">OpenAI Responses</option><option value="openai_chat">OpenAI-compatible Chat Completions</option><option value="anthropic">Anthropic Messages</option><option value="systemone">System-One decisions</option></select></label></div>
      <div className="agent-settings-connection-fields"><label>Base URL<input type="url" required value={endpoint} onChange={(event) => { setEndpoint(event.target.value); setModels([]); setModelCapabilities([]); }} /></label>
      <label>API key<input type="password" autoComplete="new-password" value={apiKey} onChange={(event) => { setApiKey(event.target.value); setModels([]); setModelCapabilities([]); }} placeholder={protocol === 'openai_chat' || protocol === 'systemone' ? 'Optional for local hosts' : 'Required'} /></label></div>
      <div className="agent-settings-model-row"><AgentModelPicker label="Default model" models={models} value={model} onChange={setModel} /><div className="agent-settings-actions"><button className="icon-button" type="button" aria-label="Detect models" title="Detect models" onClick={() => void test(false)}><Search size={16} /></button><button className="command-button" type="button" disabled={!model.trim()} onClick={() => void test(true)}>Test model</button></div></div>
      {probe && <p role="status" className={probe.connected ? 'agent-probe-ok' : 'agent-error'}>{probe.detail}</p>}
      <ModelCapabilitiesEditor model={model} capabilities={modelCapabilities} onChange={(field, value) => setModelCapabilities((current) => setCapability(current, model, field, value))} />
      <div className="agent-settings-footer"><button className="command-button primary" type="submit" disabled={!name.trim() || !model.trim() || !endpoint.trim()}><Plus size={14} />Add connection</button></div>
    </fieldset>
    {error && <p className="agent-error" role="alert">{error}</p>}
  </form>;
}

function AgentSettingsDialog({ onClose }: { readonly onClose: () => void }) {
  const [providers, setProviders] = useState<readonly AgentProviderSettings[]>([]);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const dialog = useRef<HTMLElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetchAgentProviderSettings(controller.signal).then((items) => {
      if (controller.signal.aborted) return;
      setProviders(items); setSelected(items[0]?.providerId ?? 'new');
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog.current?.querySelector('button')?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); onClose(); }
      if (event.key !== 'Tab') return;
      const elements = dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled)');
      const first = elements?.[0];
      const last = elements?.[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', keydown);
    return () => { document.removeEventListener('keydown', keydown); previous?.focus(); };
  }, [onClose]);

  const provider = providers.find((item) => item.providerId === selected);
  return createPortal(<div className="schema-dialog-backdrop">
    <section ref={dialog} className="agent-settings-dialog" role="dialog" aria-modal="true" aria-label="Agent settings">
      <header><div><h2>Agent settings</h2><p>Manage model connections</p></div><button className="icon-button" type="button" aria-label="Close agent settings" onClick={onClose}><X size={18} /></button></header>
      {error ? <p className="agent-error" role="alert">{error}</p> : <>
        <div className="agent-settings-picker"><label className="agent-settings-select">Provider<select value={selected} onChange={(event) => setSelected(event.target.value)}><option value="new">New connection</option>{providers.map((item) => <option key={item.providerId} value={item.providerId}>{item.displayName}{item.configured ? ' · configured' : ''}</option>)}</select></label><button className="icon-button" type="button" title="Add connection" aria-label="New provider connection" onClick={() => setSelected('new')}><Plus size={16} /></button></div>
        {selected === 'new' ? <NewConnectionForm onCreated={(next) => { setProviders((items) => [...items, next]); setSelected(next.providerId); }} /> : provider && <ProviderForm key={provider.providerId} settings={provider} onSaved={(next) => setProviders((items) => items.map((item) => item.providerId === next.providerId ? next : item))} onDeleted={(id) => { setProviders((items) => items.filter((item) => item.providerId !== id)); setSelected(providers.find((item) => item.providerId !== id)?.providerId ?? 'new'); }} />}
      </>}
    </section>
  </div>, document.body);
}

export function AgentSettingsButton({ compact = false }: { readonly compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  return <>
    <button className={compact ? 'icon-button' : 'command-button'} type="button" aria-label={compact ? 'Settings' : 'Configure agent providers'} title="Agent settings" onClick={() => setOpen(true)}><Settings2 size={16} />{!compact && 'Configure providers'}</button>
    {open && <AgentSettingsDialog onClose={close} />}
  </>;
}
