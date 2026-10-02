import { ArrowRight, Clock3, Folder, Package, Play, RefreshCw, Wrench } from 'lucide-react';
import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { cancelToolJob, fetchExtensions, fetchExtensionTools, fetchToolJobs, runExtensionTool } from '../api/client';
import type { ExtensionStatus, ToolView, ToolJob, JsonValue } from '../api/contracts.gen';

export function ExtensionToolsPanel() {
  const [tools, setTools] = useState<readonly ToolView[]>([]);
  const [extensions, setExtensions] = useState<readonly ExtensionStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAllJobs, setShowAllJobs] = useState(false);
  const [jobs, setJobs] = useState<readonly ToolJob[]>([]);
  const [selected, setSelected] = useState('');
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [nextTools, nextJobs, nextExtensions] = await Promise.all([fetchExtensionTools(signal), fetchToolJobs(signal), fetchExtensions(signal)]);
      if (signal?.aborted) return;
      setTools(nextTools); setJobs(nextJobs); setExtensions(nextExtensions); setError('');
    } catch (reason: unknown) {
      if (!signal?.aborted) setError(reason instanceof Error ? reason.message : 'Unable to load tools');
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);
  useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, [load]);
  const running = jobs.some((job) => job.status === 'queued' || job.status === 'running');
  useEffect(() => {
    if (!running) return;
    const controller = new AbortController();
    let pending = false;
    const timer = window.setInterval(() => {
      if (pending) return;
      pending = true;
      void load(controller.signal).finally(() => { pending = false; });
    }, 1000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [running, load]);
  const tool = tools.find((item) => `${item.extensionId}/${item.toolId}` === selected);
  const choose = useCallback((next: ToolView | undefined) => {
    setSelected(next ? `${next.extensionId}/${next.toolId}` : '');
    setConfirmed(false);
    setValues(Object.fromEntries((next?.fields ?? []).map((field) => [field.name,
      field.kind === 'boolean' ? field.default === true : field.default == null ? '' : String(field.default)])));
  }, []);
  useEffect(() => {
    if (!tools.some((item) => `${item.extensionId}/${item.toolId}` === selected)) choose(tools[0]);
  }, [tools, selected, choose]);
  const extensionIds = [...new Set(tools.map((item) => item.extensionId))];
  const visibleJobs = showAllJobs || !tool ? jobs : jobs.filter((job) => job.extensionId === tool.extensionId && job.toolId === tool.toolId);
  const activeJob = jobs.find((job) => job.extensionId === tool?.extensionId && job.toolId === tool?.toolId && (job.status === 'queued' || job.status === 'running'));
  const extensionRunning = jobs.some((job) => job.extensionId === tool?.extensionId && (job.status === 'queued' || job.status === 'running') &&
    (job.toolId === tool?.toolId || !tool?.allowConcurrent || !tools.find((item) => item.extensionId === job.extensionId && item.toolId === job.toolId)?.allowConcurrent));
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!tool) return;
    setBusy(true); setError('');
    try {
      const arguments_: Record<string, JsonValue> = {};
      for (const field of tool.fields) {
        const value = values[field.name];
        if (value === '' || value === undefined) continue;
        if (field.kind === 'integer' || field.kind === 'number') {
          const number = Number(value);
          if (!Number.isFinite(number) || (field.kind === 'integer' && !Number.isInteger(number))) throw new Error(`Invalid ${field.label}`);
          arguments_[field.name] = number;
        } else arguments_[field.name] = value;
      }
      const job = await runExtensionTool(tool.extensionId, tool.toolId, arguments_, confirmed);
      setJobs((current) => [job, ...current.filter((item) => item.jobId !== job.jobId)]);
      setConfirmed(false);
    } catch (reason: unknown) { setError(reason instanceof Error ? reason.message : 'Tool execution failed'); }
    finally { setBusy(false); }
  };
  const cancel = async (jobId: string) => {
    try { const job = await cancelToolJob(jobId); setJobs((current) => current.map((item) => item.jobId === jobId ? job : item)); }
    catch (reason: unknown) { setError(reason instanceof Error ? reason.message : 'Unable to cancel task'); }
  };
  return <section className="tools-workspace" aria-label="Tools">
    <header className="tools-toolbar">
      <span><Wrench size={15} />{tools.length} {tools.length === 1 ? 'tool' : 'tools'} <span className="tools-muted">from {extensionIds.length} {extensionIds.length === 1 ? 'extension' : 'extensions'}</span></span>
      <div className="button-row">
        <a className="command-button" href="?view=extensions"><Package size={14} />Manage extensions<ArrowRight size={13} /></a>
        <button className="icon-button bordered" type="button" onClick={() => void load()} disabled={busy} aria-label="Refresh tools" title="Refresh tools"><RefreshCw size={15} /></button>
      </div>
    </header>
    {error && <div className="tools-error" role="alert">{error}</div>}
    <div className="tools-layout">
      <nav className="tools-catalog" aria-label="Tool catalog">
        {extensionIds.map((extensionId) => {
          const extension = extensions.find((item) => item.extensionId === extensionId);
          const items = tools.filter((item) => item.extensionId === extensionId);
          return <section className="tools-group" key={extensionId} aria-label={extension?.name ?? extensionId}>
            <header><Folder size={14} /><h2>{extension?.name ?? extensionId}</h2><span>{items.length}</span></header>
            {items.map((item) => <button className={`tool-item ${selected === `${item.extensionId}/${item.toolId}` ? 'tool-item-active' : ''}`} type="button" key={item.toolId}
              aria-pressed={selected === `${item.extensionId}/${item.toolId}`} onClick={() => { choose(item); setError(''); }}>
              <Wrench size={15} /><span>{item.name}</span>
            </button>)}
          </section>;
        })}
        {tools.length === 0 && <p className="tools-catalog-empty">{loading ? 'Loading tools…' : 'No tool extensions enabled.'}</p>}
      </nav>
      <div className="tools-detail">
        {tool ? <>
          <header className="tool-heading"><div className="tool-heading-icon"><Wrench size={23} /></div><div><span className="tool-provider">{extensions.find((item) => item.extensionId === tool.extensionId)?.name ?? tool.extensionId}</span><h2>{tool.name}</h2><p>{tool.description}</p></div></header>
          <form className="tool-form" onSubmit={(event) => void submit(event)}>
            <header className="tool-section-heading"><h3>Inputs</h3><span>{tool.fields.length === 0 ? 'No configuration needed' : 'Configure this task'}</span></header>
            <div className="tool-fields">
              {tool.fields.map((field) => <label className={field.kind === 'boolean' ? 'tool-field tool-field-boolean' : 'tool-field'} key={field.name}>
                <span>{field.label}{field.required && field.kind !== 'boolean' && <small>Required</small>}</span>
                {field.kind === 'boolean' ?
                  <input aria-label={field.label} type="checkbox" disabled={busy || extensionRunning} checked={values[field.name] === true} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.checked }))} /> :
                  (field.choices?.length ?? 0) > 0 ? <select aria-label={field.label} className="plain-input" disabled={busy || extensionRunning} required={field.required} value={String(values[field.name] ?? '')} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.value }))}>
                    <option value="">Select</option>{field.choices?.map((choice) => <option key={choice}>{choice}</option>)}
                  </select> : <input aria-label={field.label} className="plain-input" disabled={busy || extensionRunning} required={field.required} type={field.kind === 'integer' || field.kind === 'number' ? 'number' : 'text'} step={field.kind === 'integer' ? 1 : 'any'} value={String(values[field.name] ?? '')} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.value }))} />}
              </label>)}
            </div>
            <footer className="tool-form-footer">
              {tool.requiresConfirmation && <label className="tool-confirm"><input type="checkbox" checked={confirmed} disabled={busy || extensionRunning} onChange={(event) => setConfirmed(event.target.checked)} />I confirm execution of this tool with these inputs.</label>}
              {activeJob && <button className="command-button" type="button" onClick={() => void cancel(activeJob.jobId)}>Stop</button>}
              <button className="command-button primary" type="submit" disabled={busy || extensionRunning || (tool.requiresConfirmation && !confirmed)}><Play size={14} />{busy ? 'Starting…' : extensionRunning ? 'Task running' : 'Run tool'}</button>
            </footer>
          </form>
        </> : <div className="tools-empty"><div className="tool-heading-icon"><Wrench size={28} /></div><h2>{loading ? 'Loading tools…' : 'Your tools will appear here'}</h2><p>{loading ? 'Loading installed extension tools.' : 'No tools are installed and enabled.'}</p>{!loading && <a className="command-button" href="?view=extensions"><Package size={14} />Manage extensions<ArrowRight size={14} /></a>}</div>}
        {(tool || jobs.length > 0) && <section className="tool-history" aria-label="Task history">
          <header className="tool-section-heading"><h3><Clock3 size={15} />Task history</h3><label className="tool-history-filter"><input type="checkbox" checked={showAllJobs} onChange={(event) => setShowAllJobs(event.target.checked)} />All tools</label></header>
          {visibleJobs.length === 0 && <p className="tool-history-empty">No tasks yet.</p>}
          {visibleJobs.map((job) => <article className="tool-job" key={job.jobId}>
            <header><strong>{tools.find((item) => item.extensionId === job.extensionId && item.toolId === job.toolId)?.name ?? `${job.extensionId}/${job.toolId}`}</strong><span className={`tool-job-status tool-job-status-${job.status}`}>{job.status === 'cancelled' ? 'stopped' : job.status}</span><time dateTime={job.createdAt}>{new Date(job.createdAt).toLocaleString()}</time>
              {(job.status === 'queued' || job.status === 'running') && <button className="command-button" type="button" onClick={() => void cancel(job.jobId)}>Stop</button>}
            </header>
            {job.result && <p>{job.result.message}</p>}
            {job.error && <p className="tool-job-error" role="alert">{job.error}</p>}
            {job.result?.data != null && <pre className="tool-result">{JSON.stringify(job.result.data, null, 2)}</pre>}
            {job.log && <details className="tool-job-log"><summary>Task log</summary><pre>{job.log}</pre></details>}
          </article>)}
        </section>}
      </div>
    </div>
  </section>;
}
