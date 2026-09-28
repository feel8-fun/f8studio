import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import type { AgentSession } from '../api/contracts';
import { AgentWorkspace } from './AgentWorkspace';

const api = vi.hoisted(() => ({
  cancelAgentRun: vi.fn(),
  createAgentSession: vi.fn(),
  fetchAgentProviders: vi.fn(),
  fetchAgentSession: vi.fn(),
  fetchAgentSessions: vi.fn(),
  fetchProjects: vi.fn(),
  resolveAgentApproval: vi.fn(),
  startAgentRun: vi.fn(),
}));

vi.mock('../api/client', () => api);

class FakeWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  close() {}
}

const baseSession: AgentSession = {
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
  vi.stubGlobal('WebSocket', FakeWebSocket);
  api.fetchProjects.mockResolvedValue([{ projectId: 'project1', name: 'Project', description: '', createdAt: '', updatedAt: '', graphRevision: 0, layoutRevision: 0 }]);
  api.fetchAgentProviders.mockResolvedValue([{ providerId: 'deterministic', displayName: 'Deterministic graph agent', models: ['graph-builder-v1'], configured: true, deterministic: true }]);
  api.fetchAgentSessions.mockResolvedValue([]);
  api.createAgentSession.mockResolvedValue(baseSession);
  api.fetchAgentSession.mockResolvedValue(baseSession);
  api.cancelAgentRun.mockResolvedValue({ ...baseSession, status: 'cancelled' });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

test('shows exact tool approval and submits its argument hash', async () => {
  const waiting: AgentSession = {
    ...baseSession,
    status: 'waiting_for_approval',
    messages: [{ messageId: 'message1', role: 'user', content: 'Build graph', createdAt: '' }],
    toolCalls: [{
      toolCallId: 'tool1', toolName: 'graph.apply_patch', arguments: {}, argumentsHash: 'hash1',
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
  expect(screen.getByText(/Tool call tool1/)).toBeInTheDocument();
  expect(screen.getByText('hash1')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Approve agent tool' }));

  await waitFor(() => expect(api.resolveAgentApproval).toHaveBeenCalledWith('session1', 'approval1', 'hash1', true));
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
  expect(await screen.findByText('cancelled')).toBeInTheDocument();
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
  await waitFor(() => expect(api.fetchAgentSession).toHaveBeenCalledWith('restored'));
});
