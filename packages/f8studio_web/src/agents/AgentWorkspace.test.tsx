import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import type { AgentSession } from '../api/contracts';
import { AgentWorkspace } from './AgentWorkspace';

const api = vi.hoisted(() => ({
  cancelAgentRun: vi.fn(),
  createAgentSession: vi.fn(),
  deleteAgentSession: vi.fn(),
  fetchAgentProviders: vi.fn(),
  fetchAgentSession: vi.fn(),
  fetchAgentSessions: vi.fn(),
  fetchProjects: vi.fn(),
  renameAgentSession: vi.fn(),
  selectAgentModel: vi.fn(),
  resolveAgentApproval: vi.fn(),
  startAgentRun: vi.fn(),
}));

vi.mock('../api/client', () => api);

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  constructor() { FakeWebSocket.instances.push(this); }
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  close() {}
}

const baseSession: AgentSession = {
  autoTitlePending: false,
  sessionId: 'session1',
  projectId: 'project1',
  title: 'Graph agent',
  providerId: 'deterministic',
  modelId: 'graph-builder-v1',
  status: 'idle',
  createdAt: '2026-09-23T00:00:00.000Z',
  updatedAt: '2026-09-23T00:00:00.000Z',
  messages: [],
  toolCalls: [],
  artifacts: [],
  approval: null,
  errorMessage: '',
  tracebackId: '',
};

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal('WebSocket', FakeWebSocket);
  api.fetchProjects.mockResolvedValue([{ projectId: 'project1', name: 'Project', description: '', createdAt: '', updatedAt: '', graphRevision: 0, layoutRevision: 0 }]);
  api.fetchAgentProviders.mockResolvedValue([{ providerId: 'deterministic', displayName: 'Deterministic graph agent', models: ['graph-builder-v1'], configured: true, deterministic: true }]);
  api.fetchAgentSessions.mockResolvedValue([]);
  api.createAgentSession.mockResolvedValue(baseSession);
  api.fetchAgentSession.mockResolvedValue(baseSession);
  api.cancelAgentRun.mockResolvedValue({ ...baseSession, status: 'cancelled' });
  api.deleteAgentSession.mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

test('reconnects after the event socket closes and refreshes the selected session', async () => {
  api.fetchAgentSessions.mockResolvedValue([baseSession]);
  render(<AgentWorkspace projectId="project1" initialSessionId="session1" />);
  await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
  vi.useFakeTimers();
  act(() => FakeWebSocket.instances[0]?.onclose?.());
  await act(async () => { await vi.advanceTimersByTimeAsync(250); });
  expect(FakeWebSocket.instances).toHaveLength(2);
  const count = api.fetchAgentSession.mock.calls.length;
  await act(async () => { FakeWebSocket.instances[1]?.onopen?.();
    FakeWebSocket.instances[1]?.onmessage?.(new MessageEvent('message', { data: JSON.stringify({ type: 'stream.hello', serverEpoch: 'reconnect', sequence: 0, resumed: false }) })); });
  expect(api.fetchAgentSession.mock.calls.length).toBeGreaterThan(count);
});

test('shows exact tool approval and submits its argument hash', async () => {
  const waiting: AgentSession = {
    ...baseSession,
    status: 'waiting_for_approval',
    messages: [{ images: [], providerId: 'deterministic', modelId: 'graph-builder-v1', messageId: 'message1', role: 'user', content: 'Build graph', createdAt: '' }],
    toolCalls: [{
      toolCallId: 'tool1', toolName: 'graph.apply_patch', arguments: { patch: { operations: [
        { op: 'createNode', node: { nodeId: 'phase', name: 'Phase 1 Hz' } },
        { op: 'connectEdge', edge: { fromNodeId: 'phase', fromPortId: 'data:output:phase', toNodeId: 'cosine', toPortId: 'data:input:phase' } },
        { op: 'deleteNode', nodeId: 'unused' },
      ] } }, argumentsHash: 'hash1',
      targetGraphRevision: 0, status: 'waiting_for_approval', createdAt: '', updatedAt: '',
      result: null, errorMessage: '', tracebackId: '',
    }],
    approval: {
      approvalId: 'approval1', toolCallId: 'tool1', toolName: 'graph.apply_patch',
      argumentsHash: 'hash1', targetGraphRevision: 0, expiresAt: '', status: 'pending', resolvedAt: null,
    },
  };
  api.startAgentRun.mockResolvedValue(waiting);
  api.resolveAgentApproval.mockResolvedValue({ ...waiting, status: 'running', approval: { ...waiting.approval!, status: 'approved' } });

  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  await waitFor(() => expect(api.fetchAgentSessions).toHaveBeenCalledWith('project1', expect.any(AbortSignal)));
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  fireEvent.change(await screen.findByRole('textbox', { name: 'Agent prompt' }), { target: { value: 'Build graph' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));

  expect(await screen.findByText('Approval required')).toBeInTheDocument();
  expect(screen.getByText('graph.apply_patch')).toBeInTheDocument();
  expect(screen.getByText('1 node · 1 connection · 1 other change')).toBeInTheDocument();
  fireEvent.click(screen.getByText('1 node · 1 connection · 1 other change'));
  expect(screen.getByText('Add Phase 1 Hz')).toBeVisible();
  expect(screen.getByText('phase.phase -> cosine.phase')).toBeVisible();
  expect(screen.getByText('Delete unused')).toBeVisible();
  expect(screen.getByText('hash1')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Approve agent tool' }));

  await waitFor(() => expect(api.resolveAgentApproval).toHaveBeenCalledWith('session1', 'approval1', 'hash1', true));
});

test('renames and deletes a selected session without touching the project', async () => {
  api.renameAgentSession.mockResolvedValue({ ...baseSession, title: 'Cosine graph' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Rename session' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Session title' }), { target: { value: 'Cosine graph' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save session title' }));
  await waitFor(() => expect(api.renameAgentSession).toHaveBeenCalledWith('session1', 'Cosine graph'));
  expect(await screen.findByText('Cosine graph', { selector: '.agent-current-title' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Delete session' }));
  expect(screen.getByRole('dialog', { name: 'Delete agent session' })).toHaveTextContent('The project graph will stay unchanged.');
  fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete agent session' })).getByRole('button', { name: 'Delete session' }));
  await waitFor(() => expect(api.deleteAgentSession).toHaveBeenCalledWith('session1'));
  expect(await screen.findByText('Create a session for the selected project.')).toBeInTheDocument();
  expect(new URL(window.location.href).searchParams.has('session')).toBe(false);
});

test('cancels an active run from the workspace', async () => {
  const running: AgentSession = { ...baseSession, status: 'running' };
  api.startAgentRun.mockResolvedValue(running);
  api.cancelAgentRun.mockResolvedValue({ ...running, status: 'cancelled' });

  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  await waitFor(() => expect(api.fetchAgentSessions).toHaveBeenCalledWith('project1', expect.any(AbortSignal)));
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  fireEvent.change(await screen.findByRole('textbox', { name: 'Agent prompt' }), { target: { value: 'Inspect graph' } });
  fireEvent.click(await screen.findByRole('button', { name: 'Run' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Cancel agent run' }));

  await waitFor(() => expect(api.cancelAgentRun).toHaveBeenCalledWith('session1'));
  expect(await screen.findByText('cancelled', { selector: '.agent-status' })).toBeInTheDocument();
});

test('prefers a configured model provider for new sessions', async () => {
  api.fetchAgentProviders.mockResolvedValue([
    { providerId: 'deterministic', displayName: 'Deterministic', models: ['graph-builder-v1'], configured: true, deterministic: true },
    { providerId: 'openai', displayName: 'OpenAI', models: ['configured-model'], configured: true, deterministic: false },
  ]);
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  await waitFor(() => expect(api.createAgentSession).toHaveBeenCalledWith({
    projectId: 'project1', title: 'Studio agent', providerId: 'openai', modelId: 'configured-model',
  }));
});

test('switches the model on an existing session and sends an attached image', async () => {
  api.fetchAgentProviders.mockResolvedValue([
    { providerId: 'deterministic', displayName: 'Deterministic', models: ['graph-builder-v1'], configured: true, deterministic: true },
    { providerId: 'openai', displayName: 'OpenAI', models: ['vision-model'], configured: true, deterministic: false, supportsImages: true },
  ]);
  const selected = { ...baseSession, providerId: 'openai', modelId: 'vision-model' };
  api.selectAgentModel.mockResolvedValue(selected);
  api.startAgentRun.mockResolvedValue({ ...selected, status: 'running' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  await screen.findByRole('textbox', { name: 'Agent prompt' });
  expect(screen.getByRole('button', { name: 'Attach images' })).toBeDisabled();
  fireEvent.change(screen.getByRole('combobox', { name: 'Session provider' }), { target: { value: 'openai' } });
  await waitFor(() => expect(api.selectAgentModel).toHaveBeenCalledWith('session1', 'openai', 'vision-model'));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Session model' })).toHaveValue('vision-model'));
  api.selectAgentModel.mockResolvedValue({ ...selected, modelId: 'vision-model-b' });
  fireEvent.change(screen.getByRole('combobox', { name: 'Session model' }), { target: { value: '' } });
  fireEvent.change(screen.getByRole('textbox', { name: 'Custom model ID' }), { target: { value: 'vision-model-b' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply session model' }));
  await waitFor(() => expect(api.selectAgentModel).toHaveBeenCalledWith('session1', 'openai', 'vision-model-b'));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Custom model ID' })).toHaveValue('vision-model-b'));
  const image = new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], 'graph.png', { type: 'image/png' });
  fireEvent.change(screen.getByLabelText('Choose agent images'), { target: { files: [image] } });
  expect(await screen.findByAltText('graph.png')).toBeInTheDocument();
  fireEvent.change(screen.getByRole('textbox', { name: 'Agent prompt' }), { target: { value: 'Inspect this' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  await waitFor(() => expect(api.startAgentRun).toHaveBeenCalledWith('session1', 'Inspect this', [
    { name: 'graph.png', dataUrl: expect.stringMatching(/^data:image\/png;base64,/) },
  ]));
});

test('allows images only for the selected model on a shared connection', async () => {
  api.fetchAgentProviders.mockResolvedValue([
    { providerId: 'deterministic', displayName: 'Deterministic', models: ['graph-builder-v1'], configured: true, deterministic: true },
    { providerId: 'openai', displayName: 'Work', models: ['text-model', 'vision-model'], configured: true, deterministic: false,
      supportsImages: true, modelCapabilities: [
        { modelId: 'text-model', imageInput: false, thinking: null },
        { modelId: 'vision-model', imageInput: true, thinking: true },
      ] },
  ]);
  const textSession = { ...baseSession, providerId: 'openai', modelId: 'text-model' };
  api.createAgentSession.mockResolvedValue(textSession);
  api.selectAgentModel.mockResolvedValue({ ...textSession, modelId: 'vision-model' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  expect(await screen.findByRole('button', { name: 'Attach images' })).toBeDisabled();
  fireEvent.change(screen.getByRole('combobox', { name: 'Session model' }), { target: { value: 'vision-model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply session model' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Attach images' })).toBeEnabled());
});

test('sends the selected reasoning effort only when the user chooses one', async () => {
  api.fetchAgentProviders.mockResolvedValue([
    { providerId: 'openai', displayName: 'OpenAI', models: ['reasoning-model'], configured: true, deterministic: false,
      modelCapabilities: [{ modelId: 'reasoning-model', imageInput: false, thinking: true }] },
  ]);
  const selected = { ...baseSession, providerId: 'openai', modelId: 'reasoning-model' };
  api.createAgentSession.mockResolvedValue(selected);
  // A reconnect/resync must return the same model as session creation.
  api.fetchAgentSession.mockResolvedValue(selected);
  api.startAgentRun.mockResolvedValue({ ...selected, status: 'running' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  const effort = await screen.findByRole('combobox', { name: 'Reasoning effort' });
  expect(effort).toHaveValue('auto');
  fireEvent.change(effort, { target: { value: 'high' } });
  fireEvent.change(screen.getByRole('textbox', { name: 'Agent prompt' }), { target: { value: 'Inspect graph' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  await waitFor(() => expect(api.startAgentRun).toHaveBeenCalledWith('session1', 'Inspect graph', [], 'high'));
});

test('shows all provider models in the existing session dropdown', async () => {
  const models = Array.from({ length: 27 }, (_, index) => `model-${index + 1}`);
  api.fetchAgentProviders.mockResolvedValue([
    { providerId: 'deterministic', displayName: 'Deterministic', models: ['graph-builder-v1'], configured: true, deterministic: true },
    { providerId: 'connection_many', displayName: 'Many models', models, configured: true, deterministic: false },
  ]);
  const selected = { ...baseSession, providerId: 'connection_many', modelId: 'model-1' };
  api.createAgentSession.mockResolvedValue(selected);
  api.selectAgentModel.mockResolvedValue({ ...selected, modelId: 'model-27' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  const newModel = await screen.findByRole('combobox', { name: 'Agent model' });
  expect(newModel.querySelectorAll('option')).toHaveLength(27);
  fireEvent.click(screen.getByRole('button', { name: 'New agent session' }));
  const sessionModel = await screen.findByRole('combobox', { name: 'Session model' });
  expect(sessionModel.querySelectorAll('option')).toHaveLength(28);
  fireEvent.change(sessionModel, { target: { value: 'model-27' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply session model' }));
  await waitFor(() => expect(api.selectAgentModel).toHaveBeenCalledWith('session1', 'connection_many', 'model-27'));
});

test('keeps the window bound to its project and restores the requested session', async () => {
  const restored: AgentSession = { ...baseSession, sessionId: 'restored', projectId: 'project2' };
  api.fetchProjects.mockResolvedValue([
    { projectId: 'project1', name: 'First', description: '', createdAt: '', updatedAt: '', graphRevision: 0, layoutRevision: 0 },
    { projectId: 'project2', name: 'Second', description: '', createdAt: '', updatedAt: '', graphRevision: 0, layoutRevision: 0 },
  ]);
  api.fetchAgentSessions.mockResolvedValue([
    { sessionId: 'other', projectId: 'project2', title: 'Other', providerId: 'deterministic', modelId: 'graph-builder-v1', status: 'idle', updatedAt: '', messageCount: 0 },
    { sessionId: 'restored', projectId: 'project2', title: 'Restored', providerId: 'deterministic', modelId: 'graph-builder-v1', status: 'idle', updatedAt: '', messageCount: 0 },
  ]);
  api.fetchAgentSession.mockResolvedValue(restored);

  render(<AgentWorkspace projectId="project2" initialSessionId="restored" />);
  await waitFor(() => expect(api.fetchAgentSessions).toHaveBeenCalledWith('project2', expect.any(AbortSignal)));
  expect(await screen.findByTitle('Second')).toBeInTheDocument();
  expect(screen.queryByRole('combobox', { name: 'Agent project' })).not.toBeInTheDocument();
  await waitFor(() => expect(api.fetchAgentSession).toHaveBeenCalledWith('restored', expect.any(AbortSignal)));
});

test('keeps the most recently selected session when requests finish out of order', async () => {
  api.fetchAgentSessions.mockResolvedValue([
    { ...baseSession, title: 'First session', messageCount: 0 },
    { ...baseSession, sessionId: 'session2', title: 'Second session', messageCount: 0 },
  ]);
  let finishFirst!: (session: AgentSession) => void;
  api.fetchAgentSession.mockImplementation((id: string) => id === 'session1'
    ? new Promise<AgentSession>((resolve) => { finishFirst = resolve; })
    : Promise.resolve({ ...baseSession, sessionId: 'session2', modelId: 'second-model' }));
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: /Second session/ }));
  expect(await screen.findByText('second-model')).toBeInTheDocument();
  await act(async () => finishFirst(baseSession));
  expect(screen.getByText('second-model')).toBeInTheDocument();
  expect(new URL(window.location.href).searchParams.get('session')).toBe('session2');
});

test('reports session loading failures in the workspace', async () => {
  api.fetchAgentSessions.mockResolvedValue([{ ...baseSession, messageCount: 0 }]);
  api.fetchAgentSession.mockRejectedValue(new Error('Session unavailable'));
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Session unavailable');
});

test('coalesces event bursts and fetches again after an in-flight refresh', async () => {
  api.fetchAgentSession.mockResolvedValue(baseSession);
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  await screen.findByRole('textbox', { name: 'Agent prompt' });
  let finishRefresh!: (session: AgentSession) => void;
  api.fetchAgentSession.mockImplementationOnce(() => new Promise<AgentSession>((resolve) => { finishRefresh = resolve; }));
  const socket = FakeWebSocket.instances.at(-1)!;
  act(() => {
    socket.onmessage?.(new MessageEvent('message', { data: JSON.stringify({ type: 'stream.hello', serverEpoch: 'burst', sequence: 0, resumed: false }) }));
    for (let index = 1; index <= 20; index += 1) socket.onmessage?.(new MessageEvent('message', { data: JSON.stringify({
      type: 'agent.session.updated', serverEpoch: 'burst', sequence: index, scope: 'project:project1', payload: { sessionId: 'session1' },
    }) }));
  });
  expect(api.fetchAgentSession).toHaveBeenCalledTimes(1);
  await act(async () => finishRefresh(baseSession));
  expect(api.fetchAgentSession).toHaveBeenCalledTimes(2);
});

test('clears the prompt only after a successful submission', async () => {
  api.startAgentRun.mockRejectedValueOnce(new Error('Model unavailable')).mockResolvedValueOnce({ ...baseSession, status: 'running' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  const prompt = await screen.findByRole('textbox', { name: 'Agent prompt' });
  fireEvent.change(prompt, { target: { value: 'Inspect graph' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Model unavailable');
  expect(prompt).toHaveValue('Inspect graph');
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  await waitFor(() => expect(prompt).toHaveValue(''));
});

test('sends with Ctrl or Command Enter, preserving newlines and IME composition', async () => {
  api.startAgentRun.mockResolvedValue(baseSession);
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  const prompt = await screen.findByRole('textbox', { name: 'Agent prompt' });
  fireEvent.change(prompt, { target: { value: 'Inspect graph' } });
  fireEvent.keyDown(prompt, { key: 'Enter' });
  fireEvent.keyDown(prompt, { key: 'Enter', ctrlKey: true, isComposing: true });
  expect(api.startAgentRun).not.toHaveBeenCalled();
  fireEvent.keyDown(prompt, { key: 'Enter', ctrlKey: true });
  await waitFor(() => expect(prompt).toHaveValue(''));
  expect(api.startAgentRun).toHaveBeenCalledWith('session1', 'Inspect graph', []);
  expect(prompt).toHaveFocus();
  fireEvent.change(prompt, { target: { value: 'Explain the nodes' } });
  fireEvent.keyDown(prompt, { key: 'Enter', metaKey: true });
  await waitFor(() => expect(api.startAgentRun).toHaveBeenCalledWith('session1', 'Explain the nodes', []));
});

test('shows tool activity between the request and response with expandable details', async () => {
  api.createAgentSession.mockResolvedValue({
    ...baseSession,
    messages: [
      { images: [], providerId: 'deterministic', modelId: 'graph-builder-v1', messageId: 'request', role: 'user', content: 'Inspect graph', createdAt: '2026-09-23T00:00:01Z' },
      { images: [], providerId: 'deterministic', modelId: 'graph-builder-v1', messageId: 'reply', role: 'assistant', content: 'Graph inspected', createdAt: '2026-09-23T00:00:03Z' },
    ],
    toolCalls: [{ toolCallId: 'inspect', toolName: 'graph.read', arguments: { projectId: 'project1' }, argumentsHash: '', targetGraphRevision: 0, status: 'succeeded', createdAt: '2026-09-23T00:00:02Z', updatedAt: '', result: { nodeCount: 2 }, errorMessage: '', tracebackId: '' }],
  });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  await screen.findByText('Graph inspected');
  const entries = screen.getAllByRole('article');
  expect(entries[0]).toHaveTextContent('Inspect graph');
  expect(entries[1]).toHaveTextContent('graph.read');
  expect(entries[2]).toHaveTextContent('Graph inspected');
  fireEvent.click(screen.getByText('Tool details'));
  expect(screen.getByText(/"nodeCount": 2/)).toBeVisible();
});

test('collapses completed tool bursts while keeping their details available', async () => {
  api.createAgentSession.mockResolvedValue({
    ...baseSession,
    messages: [{ images: [], providerId: 'deterministic', modelId: 'graph-builder-v1', messageId: 'request', role: 'user', content: 'Inspect graph', createdAt: '2026-09-23T00:00:01Z' }],
    toolCalls: ['graph.read', 'catalog.search', 'catalog.operator'].map((toolName, index) => ({
      toolCallId: `tool${index}`, toolName, arguments: {}, argumentsHash: '', targetGraphRevision: 0,
      status: 'succeeded' as const, createdAt: `2026-09-23T00:00:0${index + 2}Z`, updatedAt: '',
      result: null, errorMessage: '', tracebackId: '',
    })),
  });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  expect(await screen.findByText('3 tool calls')).toBeInTheDocument();
  const group = screen.getByText('3 tool calls').closest('details');
  expect(group).not.toHaveAttribute('open');
  fireEvent.click(screen.getByText('3 tool calls'));
  expect(group).toHaveAttribute('open');
  expect(screen.getByText('graph.read')).toBeVisible();
});

test('drops selected reasoning effort when refreshed model capabilities disable it', async () => {
  const provider = { providerId: 'connection_work', displayName: 'Work', models: ['model'], configured: true, deterministic: false };
  api.fetchAgentProviders.mockResolvedValue([
    { ...provider, modelCapabilities: [{ modelId: 'model', thinking: true }] },
  ]);
  const selected = { ...baseSession, providerId: provider.providerId, modelId: 'model' };
  api.createAgentSession.mockResolvedValue(selected);
  api.startAgentRun.mockResolvedValue({ ...selected, status: 'running' });
  render(<AgentWorkspace projectId="project1" initialSessionId={null} />);
  fireEvent.click(await screen.findByRole('button', { name: 'New agent session' }));
  fireEvent.change(await screen.findByRole('combobox', { name: 'Reasoning effort' }), { target: { value: 'high' } });
  api.fetchAgentProviders.mockResolvedValue([
    { ...provider, modelCapabilities: [{ modelId: 'model', thinking: false }] },
  ]);
  act(() => window.dispatchEvent(new Event('studio-agent-providers-changed')));
  await waitFor(() => expect(screen.queryByRole('combobox', { name: 'Reasoning effort' })).not.toBeInTheDocument());
  fireEvent.change(screen.getByRole('textbox', { name: 'Agent prompt' }), { target: { value: 'Inspect graph' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  await waitFor(() => expect(api.startAgentRun).toHaveBeenCalledWith('session1', 'Inspect graph', []));
});
