import { isJsonObject } from '../api/contracts';
import { studioEvents } from '../api/eventStream';
import { Bot, BrainCircuit, Check, ImagePlus, Pencil, Plus, Send, ShieldCheck, Square, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  cancelAgentRun,
  createAgentSession,
  deleteAgentSession,
  fetchAgentProviders,
  fetchAgentSession,
  fetchAgentSessions,
  fetchProjects,
  renameAgentSession,
  resolveAgentApproval,
  selectAgentModel,
  startAgentRun,
} from '../api/client';
import type { AgentImage, AgentProviderSummary, AgentSession, AgentSessionSummary } from '../api/contracts';
import { AgentTimeline } from './AgentTimeline';
import { AgentPatchPreview } from './AgentPatchPreview';
import { AgentModelPicker } from './AgentModelPicker';
import { AGENT_PROVIDERS_CHANGED, AgentSettingsButton } from './AgentSettings';

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];

async function readImage(file: File): Promise<AgentImage> {
  if (!IMAGE_TYPES.includes(file.type)) throw new Error(`${file.name}: use PNG, JPEG, WebP, or GIF.`);
  if (file.size > 4 * 1024 * 1024) throw new Error(`${file.name}: image exceeds 4 MB.`);
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => typeof reader.result === 'string' ? resolve(reader.result) : reject(new Error('Image read failed'));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
  return { name: file.name, dataUrl };
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

export function AgentWorkspace({ projectId, initialSessionId }: { readonly projectId: string; readonly initialSessionId: string | null }) {
  const [projectName, setProjectName] = useState(projectId);
  const [providers, setProviders] = useState<readonly AgentProviderSummary[]>([]);
  const [sessions, setSessions] = useState<readonly AgentSessionSummary[]>([]);
  const [session, setSession] = useState<AgentSession | null>(null);
  const [providerId, setProviderId] = useState('deterministic');
  const [modelId, setModelId] = useState('graph-builder-v1');
  const [sessionModelDraft, setSessionModelDraft] = useState('');
  const [reasoningEffort, setReasoningEffort] = useState<'auto' | 'low' | 'medium' | 'high'>('auto');
  const [prompt, setPrompt] = useState('');
  const [images, setImages] = useState<readonly AgentImage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [loadingSession, setLoadingSession] = useState(false);
  const [showLatest, setShowLatest] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const transcript = useRef<HTMLDivElement | null>(null);
  const followLatest = useRef(true);
  const promptInput = useRef<HTMLTextAreaElement | null>(null);
  const imageInput = useRef<HTMLInputElement | null>(null);
  const selectedSessionId = useRef<string | null>(null);
  const selectionRequest = useRef<AbortController | null>(null);

  const provider = useMemo(
    () => providers.find((candidate) => candidate.providerId === providerId) ?? providers[0],
    [providerId, providers],
  );
  const sessionProvider = providers.find((candidate) => candidate.providerId === session?.providerId);
  const selectedModelCapability = sessionProvider?.modelCapabilities?.find((item) => item.modelId === session?.modelId);
  const supportsImages = selectedModelCapability?.imageInput === true || (sessionProvider?.modelCapabilities === undefined && sessionProvider?.supportsImages === true);
  const canSetReasoningEffort = sessionProvider !== undefined && !sessionProvider.deterministic && selectedModelCapability?.thinking !== false;

  useEffect(() => { setSessionModelDraft(session?.modelId ?? ''); }, [session?.sessionId, session?.modelId]);
  useEffect(() => { setReasoningEffort('auto'); }, [session?.sessionId, session?.providerId, session?.modelId, canSetReasoningEffort]);

  const attachImages = async (files: readonly File[]) => {
    if (!supportsImages || files.length === 0) return;
    if (images.length + files.length > 3) { setError('Attach at most 3 images.'); return; }
    try {
      const added = await Promise.all(files.map(readImage));
      setImages((current) => [...current, ...added].slice(0, 3));
      setError('');
    } catch (reason) { setError(errorText(reason)); }
  };

  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => {
      void fetchAgentProviders(controller.signal).then((items) => {
        if (controller.signal.aborted) return;
        setProviders(items);
        const next = items.find((item) => item.providerId === providerId && item.configured && !item.deterministic)
          ?? items.find((item) => item.configured && !item.deterministic) ?? items.find((item) => item.configured);
        if (next !== undefined) { setProviderId(next.providerId); setModelId(next.models[0] ?? ''); }
      }).catch((reason: unknown) => { if (!controller.signal.aborted) setError(errorText(reason)); });
    };
    window.addEventListener(AGENT_PROVIDERS_CHANGED, refresh);
    window.addEventListener('focus', refresh);
    return () => {
      controller.abort();
      window.removeEventListener(AGENT_PROVIDERS_CHANGED, refresh);
      window.removeEventListener('focus', refresh);
    };
  }, [providerId]);

  const updateSession = useCallback((next: AgentSession) => {
    if (selectedSessionId.current !== next.sessionId) return;
    setSession((current) => current?.sessionId === next.sessionId && current.updatedAt > next.updatedAt ? current : next);
    setSessions((current) => current.map((item) => item.sessionId === next.sessionId && item.updatedAt <= next.updatedAt ? sessionSummary(next) : item));
  }, []);

  const selectSession = useCallback(async (sessionId: string) => {
    selectionRequest.current?.abort();
    const controller = new AbortController();
    selectionRequest.current = controller;
    selectedSessionId.current = sessionId;
    followLatest.current = true;
    setShowLatest(false);
    setLoadingSession(true);
    setSession(null);
    setPrompt('');
    setImages([]);
    setError('');
    setEditingTitle(false);
    setConfirmDelete(false);
    try {
      const next = await fetchAgentSession(sessionId, controller.signal);
      if (!controller.signal.aborted) updateSession(next);
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorText(reason));
    } finally {
      if (!controller.signal.aborted) setLoadingSession(false);
    }
  }, [updateSession]);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([fetchProjects(controller.signal), fetchAgentProviders(controller.signal)]).then(
      ([projectList, providerList]) => {
        if (controller.signal.aborted) return;
        const selected = projectList.find((item) => item.projectId === projectId);
        if (selected === undefined) { setError('This project no longer exists.'); return; }
        setProjectName(selected.name);
        setProviders(providerList);
        const firstProvider = providerList.find((item) => item.configured && !item.deterministic)
          ?? providerList.find((item) => item.configured) ?? providerList[0];
        if (firstProvider !== undefined) {
          setProviderId(firstProvider.providerId);
          setModelId(firstProvider.models[0] ?? '');
        }
      },
      (reason: unknown) => { if (!controller.signal.aborted) setError(errorText(reason)); },
    );
    return () => controller.abort();
  }, [projectId]);

  useEffect(() => {
    const controller = new AbortController();
    fetchAgentSessions(projectId, controller.signal).then(
      (items) => {
        if (controller.signal.aborted) return;
        setSessions((current) => [...current.filter((item) => !items.some((loaded) => loaded.sessionId === item.sessionId)), ...items]);
        if (selectedSessionId.current !== null) return;
        const target = items.find((item) => item.sessionId === initialSessionId) ?? items[0];
        if (target === undefined) setSession(null);
        else void selectSession(target.sessionId);
      },
      (reason: unknown) => { if (!controller.signal.aborted) setError(errorText(reason)); },
    );
    return () => {
      controller.abort();
      selectionRequest.current?.abort();
      selectedSessionId.current = null;
    };
  }, [initialSessionId, projectId, selectSession]);

  useEffect(() => {
    if (session === null) return;
    const url = new URL(window.location.href);
    if (url.searchParams.get('session') === session.sessionId) return;
    url.searchParams.set('session', session.sessionId);
    window.history.replaceState(null, '', url);
  }, [session?.sessionId]);

  useEffect(() => {
    if (transcript.current === null) return;
    if (followLatest.current) transcript.current.scrollTop = transcript.current.scrollHeight;
    else setShowLatest(true);
  }, [session?.sessionId, session?.updatedAt]);

  useEffect(() => {
    if (!busy && session !== null && session.status !== 'running' && session.status !== 'waiting_for_approval') {
      promptInput.current?.focus();
    }
  }, [busy, session?.sessionId, session?.status]);

  useEffect(() => {
    if (session === null) return;
    const sessionId = session.sessionId;
    const controller = new AbortController();
    let refreshing = false;
    let refreshPending = false;
    const refresh = async () => {
      if (refreshing) { refreshPending = true; return; }
      refreshing = true;
      try {
        do {
          refreshPending = false;
          const next = await fetchAgentSession(sessionId, controller.signal);
          if (!controller.signal.aborted) updateSession(next);
        } while (refreshPending && !controller.signal.aborted);
      } catch (reason) {
        if (!controller.signal.aborted) setError(errorText(reason));
      } finally { refreshing = false; }
    };
    const unsubscribe = studioEvents.subscribe((envelope) => {
      const body = envelope.payload;
      if (envelope.type !== 'agent.session.updated' || !isJsonObject(body)) return;
      if (body.sessionId === sessionId) void refresh();
    }, () => void refresh());
    return () => { controller.abort(); unsubscribe(); };
  }, [updateSession, session?.sessionId]);

  const create = async () => {
    if (busy || provider === undefined || !provider.configured || !modelId) return;
    setBusy(true); setError('');
    try {
      const created = await createAgentSession({
        projectId,
        title: 'Studio agent',
        providerId: provider.providerId,
        modelId,
      });
      selectionRequest.current?.abort();
      selectedSessionId.current = created.sessionId;
      followLatest.current = true;
      setShowLatest(false);
      setLoadingSession(false);
      setSessions((current) => [sessionSummary(created), ...current]);
      setSession(created);
      setPrompt('');
      setImages([]);
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const run = async () => {
    if (busy || session === null || (!prompt.trim() && images.length === 0) || (images.length > 0 && !supportsImages) || session.status === 'running' || session.status === 'waiting_for_approval') return;
    setBusy(true); setError('');
    try {
      updateSession(await (!canSetReasoningEffort || reasoningEffort === 'auto'
        ? startAgentRun(session.sessionId, prompt.trim(), images)
        : startAgentRun(session.sessionId, prompt.trim(), images, reasoningEffort)));
      if (selectedSessionId.current === session.sessionId) { setPrompt(''); setImages([]); }
    }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const resolve = async (approved: boolean) => {
    const approval = session?.approval;
    if (session === null || approval == null || approval.status !== 'pending') return;
    setBusy(true); setError('');
    try {
      updateSession(await resolveAgentApproval(
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
    try { updateSession(await cancelAgentRun(session.sessionId)); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const saveTitle = async () => {
    if (session === null || busy) return;
    const title = titleDraft.trim();
    if (!title) { setError('Session title cannot be empty.'); return; }
    if (title === session.title) { setEditingTitle(false); return; }
    setBusy(true); setError('');
    try {
      updateSession(await renameAgentSession(session.sessionId, title));
      setEditingTitle(false);
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const changeModel = async (nextProviderId: string, nextModelId: string) => {
    if (session === null || busy || activeRun || !nextModelId) return;
    setBusy(true); setError('');
    try { updateSession(await selectAgentModel(session.sessionId, nextProviderId, nextModelId)); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const removeSession = async () => {
    if (session === null || busy) return;
    setBusy(true); setError('');
    try {
      await deleteAgentSession(session.sessionId);
      const remaining = sessions.filter((item) => item.sessionId !== session.sessionId);
      setSessions(remaining);
      selectionRequest.current?.abort();
      selectedSessionId.current = null;
      setSession(null);
      setConfirmDelete(false);
      const url = new URL(window.location.href);
      url.searchParams.delete('session');
      window.history.replaceState(null, '', url);
      if (remaining[0] !== undefined) void selectSession(remaining[0].sessionId);
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  };

  const activeRun = session?.status === 'running' || session?.status === 'waiting_for_approval';
  const approvalCall = session?.toolCalls.find((call) => call.toolCallId === session.approval?.toolCallId);

  return <div className="agent-workspace">
    <aside className="agent-sessions">
      <div className="agent-project-select">
        <strong title={projectName}>{projectName}</strong>
        <button className="icon-button" type="button" title="New session" aria-label="New agent session" disabled={busy || !provider?.configured || !modelId} onClick={() => void create()}><Plus size={17} /></button>
      </div>
      <div className="agent-provider-row">
        <select aria-label="Agent provider" value={provider?.providerId ?? ''} onChange={(event) => {
          const next = providers.find((item) => item.providerId === event.target.value);
          if (next !== undefined) { setProviderId(next.providerId); setModelId(next.models[0] ?? ''); }
        }}>
          {providers.map((item) => <option key={item.providerId} value={item.providerId} disabled={!item.configured}>{item.displayName}{item.configured ? '' : ' (not configured)'}</option>)}
        </select>
        <select aria-label="Agent model" value={modelId} onChange={(event) => setModelId(event.target.value)}>{provider?.models.map((model) => <option key={model} value={model}>{model}</option>)}</select>
        <AgentSettingsButton />
      </div>
      <div className="agent-session-list">
        {sessions.map((item) => <button type="button" className={item.sessionId === session?.sessionId ? 'selected' : ''} key={item.sessionId} disabled={busy} onClick={() => void selectSession(item.sessionId)}>
          <Bot size={15} /><span><strong>{item.title}</strong><small>{item.status.replaceAll('_', ' ')}</small></span>
        </button>)}
      </div>
    </aside>

    <section className="agent-conversation">
      {session === null ? <div className="agent-empty" role="status"><Bot size={28} /><span>{loadingSession ? 'Loading conversation…' : 'Create a session for the selected project.'}</span></div> : <>
        <div className="agent-run-header">
          {editingTitle ? <div className="agent-title-edit"><input autoFocus aria-label="Session title" maxLength={120} value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} onKeyDown={(event) => {
            if (event.key === 'Enter') { event.preventDefault(); void saveTitle(); }
            if (event.key === 'Escape') setEditingTitle(false);
          }} /><button className="icon-button" type="button" title="Save title" aria-label="Save session title" disabled={busy || !titleDraft.trim()} onClick={() => void saveTitle()}><Check size={15} /></button><button className="icon-button" type="button" title="Cancel rename" aria-label="Cancel session rename" onClick={() => setEditingTitle(false)}><X size={15} /></button></div> : <strong className="agent-current-title" title={session.title}>{session.title}</strong>}
          <span className={`agent-status status-${session.status}`}>{session.status.replaceAll('_', ' ')}</span>
          <div className="agent-active-model">
            <select aria-label="Session provider" title="Model for the next turn" value={session.providerId} disabled={busy || activeRun} onChange={(event) => {
              const next = providers.find((item) => item.providerId === event.target.value);
              if (next?.models[0]) void changeModel(next.providerId, next.models[0]);
            }}>{providers.map((item) => <option key={item.providerId} value={item.providerId} disabled={!item.configured}>{item.displayName}</option>)}</select>
            {sessionProvider?.deterministic ? <span className="agent-fixed-model" title={session.modelId}>{session.modelId}</span> : <><AgentModelPicker key={session.providerId} compact label="Session model" models={sessionProvider?.models ?? []} value={sessionModelDraft} disabled={busy || activeRun} onChange={setSessionModelDraft} onKeyDown={(event) => {
              if (event.key === 'Enter') { event.preventDefault(); void changeModel(session.providerId, sessionModelDraft.trim()); }
              if (event.key === 'Escape') setSessionModelDraft(session.modelId);
            }} /><button className="icon-button" type="button" title="Apply model for next turn" aria-label="Apply session model" disabled={busy || activeRun || !sessionModelDraft.trim() || sessionModelDraft.trim() === session.modelId} onClick={() => void changeModel(session.providerId, sessionModelDraft.trim())}><Check size={14} /></button></>}
          </div>
          {!editingTitle && <button className="icon-button" type="button" title={activeRun ? 'Stop the run before renaming' : 'Rename session'} aria-label="Rename session" disabled={busy || activeRun} onClick={() => { setTitleDraft(session.title); setEditingTitle(true); }}><Pencil size={14} /></button>}
          <button className="icon-button" type="button" title={activeRun ? 'Stop the run before deleting' : 'Delete session'} aria-label="Delete session" disabled={busy || activeRun} onClick={() => setConfirmDelete(true)}><Trash2 size={14} /></button>
          {activeRun && <button className="icon-button" type="button" title="Cancel run" aria-label="Cancel agent run" disabled={busy} onClick={() => void cancel()}><Square size={14} /></button>}
        </div>
        <div className="agent-transcript" ref={transcript} aria-label="Agent conversation" onScroll={() => {
          const element = transcript.current;
          if (element === null) return;
          followLatest.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48;
          if (followLatest.current) setShowLatest(false);
        }}>
          {session.messages.length === 0 && <div className="agent-welcome"><Bot size={24} /><strong>What would you like to build?</strong><p>Ask about this graph, describe a change, or get help with a Python node.</p><small>Graph changes and deployments will ask for your approval.</small></div>}
          <AgentTimeline session={session} />
          {session.approval?.status === 'pending' && <div className="agent-approval">
            <div className="agent-approval-heading"><ShieldCheck size={17} /><div><strong>Approval required</strong><span>{session.approval.toolName} · revision {session.approval.targetGraphRevision}</span></div></div>
            <AgentPatchPreview call={approvalCall} />
            <div className="agent-approval-actions"><small>Expires {new Date(session.approval.expiresAt).toLocaleTimeString()} · <code>{session.approval.argumentsHash}</code></small><button type="button" className="command-button" aria-label="Deny agent tool" disabled={busy} onClick={() => void resolve(false)}><X size={15} />Reject</button><button type="button" className="command-button primary" aria-label="Approve agent tool" disabled={busy} onClick={() => void resolve(true)}><Check size={15} />Approve</button></div>
          </div>}
          {session.errorMessage && <div className="agent-error">{session.errorMessage}{session.tracebackId ? ` · traceback ${session.tracebackId}` : ''}</div>}
        </div>
        <div className="agent-compose">
          {showLatest && <button type="button" className="agent-latest command-button" onClick={() => {
            followLatest.current = true;
            setShowLatest(false);
            if (transcript.current !== null) transcript.current.scrollTop = transcript.current.scrollHeight;
          }}>Jump to latest</button>}
          {images.length > 0 && <div className="agent-compose-images">{images.map((image, index) => <div key={`${image.name}-${index}`} className="agent-compose-image"><img src={image.dataUrl} alt={image.name} /><span title={image.name}>{image.name}</span><button className="icon-button" type="button" aria-label={`Remove ${image.name}`} title={`Remove ${image.name}`} onClick={() => setImages((current) => current.filter((_, position) => position !== index))}><X size={14} /></button></div>)}</div>}
          <textarea ref={promptInput} aria-label="Agent prompt" aria-describedby="agent-compose-hint" placeholder="Describe what you want to do…" rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} onPaste={(event) => {
            const files = Array.from(event.clipboardData.files).filter((file) => IMAGE_TYPES.includes(file.type));
            if (files.length > 0 && supportsImages) { event.preventDefault(); void attachImages(files); }
          }} onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.nativeEvent.isComposing) {
              event.preventDefault();
              void run();
            }
          }} disabled={busy || session.status === 'running' || session.status === 'waiting_for_approval'} />
          <div className="agent-compose-actions"><input ref={imageInput} type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple hidden aria-label="Choose agent images" onChange={(event) => { void attachImages(Array.from(event.target.files ?? [])); event.target.value = ''; }} /><button type="button" className="icon-button" aria-label="Attach images" title={supportsImages ? 'Attach images' : 'Selected model does not support images'} disabled={busy || activeRun || !supportsImages} onClick={() => imageInput.current?.click()}><ImagePlus size={17} /></button>{canSetReasoningEffort && <label className="agent-effort-control" title="Reasoning effort for this run; the selected model must support it"><BrainCircuit size={16} aria-hidden="true" /><select aria-label="Reasoning effort" value={reasoningEffort} disabled={busy || activeRun} onChange={(event) => setReasoningEffort(event.target.value as 'auto' | 'low' | 'medium' | 'high')}><option value="auto">Auto</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>}<button type="button" className="command-button primary" disabled={busy || (!prompt.trim() && images.length === 0) || (images.length > 0 && !supportsImages) || activeRun} onClick={() => void run()}><Send size={15} />Run</button></div>
          <small id="agent-compose-hint" className="agent-compose-hint">{session.status === 'running' ? 'Agent is working. You can stop it above.' : session.status === 'waiting_for_approval' ? 'Review the proposed action above to continue.' : images.length > 0 && !supportsImages ? 'Switch to an image-capable model or remove the attached images.' : 'Ctrl / ⌘ + Enter to send · Enter for a new line'}</small>
        </div>
      </>}
      {error && <div className="agent-error" role="alert">{error}</div>}
    </section>
    {confirmDelete && session !== null && <div className="agent-dialog-backdrop"><div className="agent-delete-dialog" role="dialog" aria-modal="true" aria-label="Delete agent session" onKeyDown={(event) => { if (event.key === 'Escape') setConfirmDelete(false); }}><strong>Delete session?</strong><p>Delete “{session.title}” and its conversation history? The project graph will stay unchanged.</p><div><button autoFocus type="button" className="command-button" disabled={busy} onClick={() => setConfirmDelete(false)}>Cancel</button><button type="button" className="command-button danger" disabled={busy} onClick={() => void removeSession()}><Trash2 size={15} />Delete session</button></div></div></div>}
  </div>;
}
