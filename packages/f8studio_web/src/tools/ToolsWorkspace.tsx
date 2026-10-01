import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { cancelToolJob, fetchExtensionTools, fetchToolJobs, runExtensionTool } from '../api/client';
import type { ToolView, ToolJob, JsonValue } from '../api/contracts.gen';

export function ToolsWorkspace() {
  const [tools, setTools] = useState<readonly ToolView[]>([]);
  const [jobs, setJobs] = useState<readonly ToolJob[]>([]);
  const [selected, setSelected] = useState('');
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [nextTools, nextJobs] = await Promise.all([fetchExtensionTools(signal), fetchToolJobs(signal)]);
      if (signal?.aborted) return;
      setTools(nextTools); setJobs(nextJobs); setError('');
    } catch (reason: unknown) {
      if (!signal?.aborted) setError(reason instanceof Error ? reason.message : 'Unable to load tools');
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
  const choose = (key: string) => {
    setSelected(key); setConfirmed(false); setError('');
    const next = tools.find((item) => `${item.extensionId}/${item.toolId}` === key);
    setValues(Object.fromEntries((next?.fields ?? []).map((field) => [field.name,
      field.kind === 'boolean' ? field.default === true : field.default == null ? '' : String(field.default)])));
  };
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
  return <section className="tools-workspace" style={{ padding: 20, overflow: 'auto', height: '100%' }}>
    <h2>Tools</h2>
    <p>Run tasks supplied by installed extensions. Install and enable tool packages in Services & Extensions.</p>
    <button type="button" onClick={() => void load()} disabled={busy}>Refresh</button>
    {error && <p role="alert">{error}</p>}
    {tools.length === 0 ? <p>No tools are installed and enabled.</p> : <label>Tool <select value={selected} onChange={(event) => choose(event.target.value)}>
      <option value="">Select a tool</option>
      {tools.map((item) => <option key={`${item.extensionId}/${item.toolId}`} value={`${item.extensionId}/${item.toolId}`}>{item.name} ({item.extensionId})</option>)}
    </select></label>}
    {tool && <form onSubmit={(event) => void submit(event)}>
      <p>{tool.description}</p>
      {tool.fields.map((field) => <div key={field.name}><label>{field.label}{field.kind === 'boolean' ?
        <input type="checkbox" checked={values[field.name] === true} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.checked }))} /> :
        (field.choices?.length ?? 0) > 0 ? <select required={field.required} value={String(values[field.name] ?? '')} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.value }))}>
          <option value="">Select</option>{field.choices?.map((choice) => <option key={choice}>{choice}</option>)}
        </select> : <input required={field.required} type={field.kind === 'integer' || field.kind === 'number' ? 'number' : 'text'} step={field.kind === 'integer' ? 1 : 'any'} value={String(values[field.name] ?? '')} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.value }))} />
      }</label></div>)}
      {tool.requiresConfirmation && <label><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />I confirm execution of this tool with these inputs.</label>}
      <div><button type="submit" disabled={busy || (tool.requiresConfirmation && !confirmed) || jobs.some((job) => job.extensionId === tool.extensionId && (job.status === 'queued' || job.status === 'running'))}>Run tool</button></div>
    </form>}
    <h3>Task history</h3>
    {jobs.length === 0 && <p>No tasks yet.</p>}
    {jobs.map((job) => <article key={job.jobId} style={{ borderTop: '1px solid var(--border)', padding: '12px 0' }}>
      <strong>{job.extensionId}/{job.toolId}</strong> · {job.status} · {job.createdAt}
      {(job.status === 'queued' || job.status === 'running') && <button type="button" onClick={() => void cancel(job.jobId)}>Cancel</button>}
      {job.result && <p>{job.result.message}</p>}
      {job.error && <p role="alert">{job.error}</p>}
      {job.result?.data != null && <pre>{JSON.stringify(job.result.data, null, 2)}</pre>}
      {job.log && <details><summary>Task log</summary><pre style={{ whiteSpace: 'pre-wrap' }}>{job.log}</pre></details>}
    </article>)}
  </section>;
}
