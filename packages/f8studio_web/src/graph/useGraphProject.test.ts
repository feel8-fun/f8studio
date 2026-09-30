import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import type { ProjectRecord } from '../api/contracts';
import type { StudioEvent } from '../api/eventStream';
import { useGraphProject } from './useGraphProject';

const api = vi.hoisted(() => ({
  fetchProjects: vi.fn(), fetchProject: vi.fn(), fetchLatestDeployment: vi.fn(), fetchCatalog: vi.fn(),
  patchProject: vi.fn(), changeHistory: vi.fn(), createProject: vi.fn(), deleteProject: vi.fn(),
  exportProjectGraph: vi.fn(), importProjectGraph: vi.fn(), refreshCatalog: vi.fn(),
}));
const events = vi.hoisted(() => ({ subscribe: vi.fn(), unsubscribe: vi.fn() }));
vi.mock('../api/client', () => api);
vi.mock('../api/eventStream', () => ({ studioEvents: events }));

const initial: ProjectRecord = {
  projectId: 'p', name: 'Project', description: '', createdAt: '', updatedAt: '',
  document: { schemaVersion: 'f8studio-document/2', projectId: 'p', graphId: 'p', graphRevision: 0,
    layoutRevision: 0, nodes: [], edges: [], layout: [] },
};
const reset = vi.fn();
const report = vi.fn();

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
  api.fetchProjects.mockResolvedValue([initial]);
  api.fetchProject.mockResolvedValue(initial);
  api.fetchLatestDeployment.mockResolvedValue(null);
  api.fetchCatalog.mockResolvedValue({ services: [], operators: [] });
  events.subscribe.mockReturnValue(events.unsubscribe);
});
afterEach(cleanup);

test('queued edits use the revision returned by the previous edit and clear saving', async () => {
  const { result } = renderHook(() => useGraphProject(reset, report));
  await waitFor(() => expect(result.current.project).toEqual(initial));
  const revisions: number[] = [];
  api.patchProject.mockImplementation(async (_id, document: ProjectRecord['document']) => {
    revisions.push(document.graphRevision);
    return { document: { ...document, graphRevision: document.graphRevision + 1 }, runtimeErrors: [] };
  });
  await act(async () => {
    const edit = [{ op: 'deleteNode' as const, nodeId: 'missing' }];
    await Promise.all([result.current.commit(edit), result.current.commit(edit)]);
  });
  expect(revisions).toEqual([0, 1]);
  expect(result.current.project?.document.graphRevision).toBe(2);
  expect(result.current.saving).toBe(false);
});

test('failed edit reloads authoritative state and invalidates dependent edits', async () => {
  const { result } = renderHook(() => useGraphProject(reset, report));
  await waitFor(() => expect(result.current.project).toEqual(initial));
  api.fetchProject.mockResolvedValue({ ...initial, document: { ...initial.document, graphRevision: 5 } });
  api.patchProject.mockRejectedValue(new Error('revision conflict'));
  await act(async () => {
    const edit = [{ op: 'deleteNode' as const, nodeId: 'missing' }];
    const outcomes = await Promise.allSettled([result.current.commit(edit), result.current.commit(edit)]);
    expect(outcomes.every((outcome) => outcome.status === 'rejected')).toBe(true);
  });
  expect(api.patchProject).toHaveBeenCalledTimes(1);
  expect(result.current.project?.document.graphRevision).toBe(5);
  expect(result.current.saving).toBe(false);
  expect(result.current.error).not.toBeNull();
});

test('late graph events cannot roll back a newer revision and unsubscribe on unmount', async () => {
  let receive: (event: StudioEvent) => void = () => {};
  events.subscribe.mockImplementation((listener: typeof receive) => { receive = listener; return events.unsubscribe; });
  const { result, unmount } = renderHook(() => useGraphProject(reset, report));
  await waitFor(() => expect(result.current.project).toEqual(initial));
  const send = (revision: number) => receive({
    type: 'graph.committed', scope: 'project:p', serverEpoch: 'test', sequence: revision, eventId: String(revision), timestamp: '',
    payload: { document: { ...initial.document, graphRevision: revision } },
  });
  act(() => { send(3); send(1); });
  expect(result.current.project?.document.graphRevision).toBe(3);
  unmount();
  expect(events.unsubscribe).toHaveBeenCalledTimes(1);
});
