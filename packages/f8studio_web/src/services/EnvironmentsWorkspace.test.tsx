import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { EnvironmentsWorkspace } from './EnvironmentsWorkspace';

const api = vi.hoisted(() => ({ fetchEnvironments: vi.fn(), fetchExtensions: vi.fn(), fetchRuntimeStorage: vi.fn() }));
vi.mock('../api/client', () => api);
const environment = { environmentId: 'base', name: 'Python runtime', source: 'official', runtimeKind: 'workspace',
  revision: '1234', state: 'ready', ready: true, extensionIds: ['vision', 'pose'], serviceClasses: ['f8.mp.pose'], toolIds: [], detail: '', baseEnvironmentId: null, pinned: false };

beforeEach(() => {
  api.fetchEnvironments.mockResolvedValue([environment]);
  api.fetchExtensions.mockResolvedValue([{ extensionId: 'vision', name: 'Computer Vision' }, { extensionId: 'pose', name: 'MediaPipe Pose' }]);
  api.fetchRuntimeStorage.mockResolvedValue({ path: '/data', cachePath: '/data/cache', canChange: true });
});
afterEach(() => { cleanup(); vi.resetAllMocks(); vi.useRealTimers(); });

test('loads standalone environment management and its extension references', async () => {
  render(<EnvironmentsWorkspace />);
  expect(await screen.findByRole('region', { name: 'Runtime environments' })).toBeInTheDocument();
  expect(screen.getByText('Computer Vision, MediaPipe Pose')).toBeInTheDocument();
  expect(screen.getByText('f8.mp.pose')).toBeInTheDocument();
  expect(screen.getByText('1 environments')).toBeInTheDocument();
  expect(screen.queryByText('Add extension package')).not.toBeInTheDocument();
});

test('reports load failures and supports retry', async () => {
  api.fetchEnvironments.mockRejectedValueOnce(new Error('Environment registry unavailable'));
  render(<EnvironmentsWorkspace />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Environment registry unavailable');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh runtime environments' }));
  await screen.findByRole('button', { name: 'Python runtime' });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

test('polls preparation until ready and stops polling when it completes', async () => {
  vi.useFakeTimers();
  api.fetchEnvironments.mockResolvedValueOnce([{ ...environment, state: 'preparing', ready: false }]).mockResolvedValue([environment]);
  render(<EnvironmentsWorkspace />);
  await act(async () => { await vi.advanceTimersByTimeAsync(0); });
  expect(screen.getByRole('button', { name: 'Cancel preparation' })).toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(screen.queryByRole('button', { name: 'Cancel preparation' })).not.toBeInTheDocument();
  expect(api.fetchEnvironments).toHaveBeenCalledTimes(2);
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(api.fetchEnvironments).toHaveBeenCalledTimes(2);
});
