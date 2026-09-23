import { Bot, Check, FileDiff, Plus, Send, ShieldCheck, Square, Wrench, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  cancelAgentRun,
  createAgentSession,
  fetchAgentProviders,
  fetchAgentSession,
  fetchAgentSessions,
  fetchProjects,
  resolveAgentApproval,
  startAgentRun,
} from '../api/client';
import type { AgentProviderSummary, AgentSession, AgentSessionSummary, ProjectSummary } from '../api/contracts';

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function sessionSummary(session: AgentSession): AgentSessionSummary {
  return {
    sessionId: session.sessionId,
    projectId: session.projectId,
    title: session.title,
    providerId: session.providerId,
    modelId: session.modelId,
    status: session.status,
    updatedAt: session.updatedAt,
    messageCount: session.messages.length,
  };
}

export function AgentWorkspace() {
  const [projects, setProjects] = useState<readonly ProjectSummary[]>([]);
  const [providers, setProviders] = useState<readonly AgentProviderSummary[]>([]);
  const [projectId, setProjectId] = useState('');
  const [sessions, setSessions] = useState<readonly AgentSessionSummary[]>([]);
  const [session, setSession] = useState<AgentSession | null>(null);
  const [providerId, setProviderId] = useState('deterministic');
  const [modelId, setModelId] = useState('graph-builder-v1');
  const [prompt, setPrompt] = useState('Build a controllable value graph, validate it, deploy it, and report runtime evidence.');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const provider = useMemo(
    () => providers.find((candidate) => candidate.providerId === providerId) ?? providers[0],
    [providerId, providers],
  );

  const refreshSession = useCallback(async (sessionId: string) => {
    const next = await fetchAgentSession(sessionId);
    setSession(next);
    setSessions((current) => current.map((item) => item.sessionId === next.sessionId ? sessionSummary(next) : item));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([fetchProjects(controller.signal), fetchAgentProviders(controller.signal)]).then(
      ([projectList, providerList]) => {
        setProjects(projectList);
        setProviders(providerList);
        const firstProject = projectList[0]?.projectId ?? '';
        setProjectId((current) => current || firstProject);
        const firstProvider = providerList.find((item) => item.configured) ?? providerList[0];
        if (firstProvider !== undefined) {
          setProviderId(firstProvider.providerId);
          setModelId(firstProvider.models[0] ?? '');
        }
      },
      (reason: unknown) => { if (!controller.signal.aborted) setError(errorText(reason)); },
    );
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!projectId) { setSessions([]); setSession(null); return; }
    const controller = new AbortController();
    fetchAgentSessions(projectId, controller.signal).then(
      (items) => {
        setSessions(items);
        const target = items.find((item) => item.sessionId === session?.sessionId) ?? items[0];
        if (target === undefined) setSession(null);
        else void refreshSession(target.sessionId).catch((reason: unknown) => setError(errorText(reason)));
      },
      (reason: unknown) => { if (!controller.signal.aborted) setError(errorText(reason)); },
    );
    return () => controller.abort();
  }, [projectId, refreshSession]);

  useEffect(() => {
    if (session === null) return;
    const sessionId = session.sessionId;
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/events`);
    socket.onmessage = (event) => {
      let payload: unknown;
      try { payload = JSON.parse(String(event.data)); }
      catch (reason) { console.error('Invalid agent event JSON', reason); return; }
      if (typeof payload !== 'object' || payload === null) return;
      const envelope = payload as Record<string, unknown>;
      const body = envelope.payload;
      if (envelope.type !== 'agent.session.updated' || typeof body !== 'object' || body === null) return;
      if ((body as Record<string, unknown>).sessionId !== sessionId) return;
      void refreshSession(sessionId).catch((reason: unknown) => setError(errorText(reason)));
    };
    socket.onerror = () => setError('Agent event stream disconnected');
    return () => socket.close();
  }, [refreshSession, session?.sessionId]);

  const create = async () => {
    if (!projectId || provider === undefined || !modelId) return;
    setBusy(true); setError('');
    try {
      const created = await createAgentSession({
        projectId,
        title: 'Graph agent',
        providerId: provider.providerId,
        modelId,
      });
      setSessions((current) => [sessionSummary(created), ...current]);
      setSession(created);
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const run = async () => {
    if (session === null || !prompt.trim()) return;
    setBusy(true); setError('');
    try { setSession(await startAgentRun(session.sessionId, prompt.trim())); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const resolve = async (approved: boolean) => {
    const approval = session?.approval;
    if (session === null || approval == null || approval.status !== 'pending') return;
    setBusy(true); setError('');
    try {
      setSession(await resolveAgentApproval(
        session.sessionId,
        approval.approvalId,
        approval.argumentsHash,
        approved,
      ));
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const cancel = async () => {
    if (session === null) return;
    setBusy(true); setError('');
    try { setSession(await cancelAgentRun(session.sessionId)); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  return <div className="agent-workspace">
    <aside className="agent-sessions">
      <div className="agent-project-select">
        <select aria-label="Agent project" value={projectId} onChange={(event) => setProjectId(event.target.value)}>
          {projects.map((project) => <option key={project.projectId} value={project.projectId}>{project.name}</option>)}
        </select>
        <button className="icon-button" type="button" title="New session" aria-label="New agent session" disabled={busy || !projectId} onClick={() => void create()}><Plus size={17} /></button>
      </div>
      <div className="agent-provider-row">
        <select aria-label="Agent provider" value={provider?.providerId ?? ''} onChange={(event) => {
          const next = providers.find((item) => item.providerId === event.target.value);
          if (next !== undefined) { setProviderId(next.providerId); setModelId(next.models[0] ?? ''); }
        }}>
          {providers.map((item) => <option key={item.providerId} value={item.providerId} disabled={!item.configured}>{item.displayName}{item.configured ? '' : ' (not configured)'}</option>)}
        </select>
        <select aria-label="Agent model" value={modelId} onChange={(event) => setModelId(event.target.value)}>{provider?.models.map((model) => <option key={model} value={model}>{model}</option>)}</select>
      </div>
      <div className="agent-session-list">
        {sessions.map((item) => <button type="button" className={item.sessionId === session?.sessionId ? 'selected' : ''} key={item.sessionId} onClick={() => void refreshSession(item.sessionId)}>
          <Bot size={15} /><span><strong>{item.title}</strong><small>{item.status.replaceAll('_', ' ')}</small></span>
        </button>)}
      </div>
    </aside>

    <section className="agent-conversation">
      {session === null ? <div className="agent-empty"><Bot size={28} /><span>Create a session for the selected project.</span></div> : <>
        <div className="agent-run-header"><span className={`agent-status status-${session.status}`}>{session.status.replaceAll('_', ' ')}</span><span>{session.providerId} / {session.modelId}</span>{(session.status === 'running' || session.status === 'waiting_for_approval') && <button className="icon-button" type="button" title="Cancel run" aria-label="Cancel agent run" disabled={busy} onClick={() => void cancel()}><Square size={14} /></button>}</div>
        <div className="agent-transcript">
          {session.messages.map((message) => <article className={`agent-message ${message.role}`} key={message.messageId}><span>{message.role}</span><p>{message.content}</p></article>)}
          {session.toolCalls.map((call) => <article className="agent-tool-call" key={call.toolCallId}><Wrench size={14} /><div><strong>{call.toolName}</strong><small>{call.status.replaceAll('_', ' ')}{call.targetGraphRevision === null ? '' : ` · revision ${call.targetGraphRevision}`}</small>{call.errorMessage && <p>{call.errorMessage}{call.tracebackId ? ` · ${call.tracebackId}` : ''}</p>}</div></article>)}
          {session.artifacts.map((artifact) => <details className="agent-artifact" key={artifact.artifactId}><summary><FileDiff size={14} />{artifact.title}</summary><pre>{JSON.stringify(artifact.payload, null, 2)}</pre></details>)}
          {session.approval?.status === 'pending' && <div className="agent-approval"><ShieldCheck size={18} /><div><strong>Approval required</strong><span>{session.approval.toolName} at graph revision {session.approval.targetGraphRevision}</span><small>Tool call {session.approval.toolCallId} · expires {new Date(session.approval.expiresAt).toLocaleTimeString()}</small><code>{session.approval.argumentsHash}</code></div><button type="button" className="icon-button approve" title="Approve" aria-label="Approve agent tool" disabled={busy} onClick={() => void resolve(true)}><Check size={18} /></button><button type="button" className="icon-button deny" title="Deny" aria-label="Deny agent tool" disabled={busy} onClick={() => void resolve(false)}><X size={18} /></button></div>}
          {session.errorMessage && <div className="agent-error">{session.errorMessage}{session.tracebackId ? ` · traceback ${session.tracebackId}` : ''}</div>}
        </div>
        <div className="agent-compose"><textarea aria-label="Agent prompt" rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} disabled={busy || session.status === 'running' || session.status === 'waiting_for_approval'} /><button type="button" className="command-button primary" disabled={busy || !prompt.trim() || session.status === 'running' || session.status === 'waiting_for_approval'} onClick={() => void run()}><Send size={15} />Run</button></div>
      </>}
      {error && <div className="agent-error" role="alert">{error}</div>}
    </section>
  </div>;
}
