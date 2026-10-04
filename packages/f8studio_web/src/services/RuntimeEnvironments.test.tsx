import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { RuntimeEnvironments } from './RuntimeEnvironments';
import type { EnvironmentStatus } from '../api/contracts.gen';

const api = vi.hoisted(() => ({ createEnvironment: vi.fn(), prepareEnvironment: vi.fn(), cancelEnvironmentPreparation: vi.fn(),
  fetchEnvironmentDetail: vi.fn(), fetchRuntimeStorage: vi.fn(), retainEnvironment: vi.fn(), removeEnvironment: vi.fn(), setRuntimeStorage: vi.fn() }));
vi.mock('../api/client', () => api);
const refresh = vi.fn().mockResolvedValue(undefined);
const environment: EnvironmentStatus = { environmentId: 'base-id', name: 'web-studio-runtime', runtimeKind: 'workspace',
  source: 'official', revision: '1234', state: 'changed', detail: 'Definition changed; verify to update records.',
  ready: true, extensionIds: ['engine'], serviceClasses: ['f8.pyengine'], toolIds: ['diagnostics/verify'], baseEnvironmentId: null, pinned: false };
const developer: EnvironmentStatus = { ...environment, environmentId: 'dev-id', name: 'my-tools', source: 'developer', state: 'declared', ready: false, detail: '', extensionIds: [], serviceClasses: [], toolIds: [] };

beforeEach(() => {
  api.fetchRuntimeStorage.mockResolvedValue({ path: '/data/runtime-storage', cachePath: '/data/runtime-storage/package-cache', canChange: true });
  api.fetchEnvironmentDetail.mockResolvedValue({ environmentId: 'dev-id', name: 'my-tools', revision: 'abcd', manifest: '[workspace]', definitionPath: '/data/definitions/my-tools/pixi.toml', sourceEnvironment: 'runtime', baseEnvironmentId: null, policy: 'preserve', condaDependencies: [], pypiDependencies: [], storagePath: '/data/envs/my-tools', cachePath: '/data/package-cache', usage: { logicalBytes: 1024, sharedLinkBytes: 512, exclusiveFileBytes: 512 }, pinned: false, changedPackages: [] });
});
afterEach(() => { cleanup(); vi.resetAllMocks(); refresh.mockResolvedValue(undefined); });

test('shows named environments, distinct changed state and service/tool references', async () => {
  render(<RuntimeEnvironments environments={[environment]} extensions={[]} onRefresh={refresh} locked={false} />);
  expect(screen.getByRole('button', { name: 'web-studio-runtime' })).toBeInTheDocument();
  expect(screen.getByText('Definition changed')).toBeInTheDocument();
  expect(screen.getByText('f8.pyengine')).toBeInTheDocument();
  expect(screen.getByText('diagnostics/verify')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Verify and update' }));
  expect(api.prepareEnvironment).toHaveBeenCalledWith('base-id');
  await screen.findByDisplayValue('/data/runtime-storage');
});

test('creates a lazy derived revision with explicit requirements and adjustment policy', async () => {
  api.createEnvironment.mockResolvedValue(developer);
  render(<RuntimeEnvironments environments={[environment]} extensions={[]} onRefresh={refresh} locked={false} />);
  fireEvent.click(screen.getByText('Create developer environment'));
  fireEvent.change(screen.getByLabelText('Environment name'), { target: { value: 'my-tools' } });
  fireEvent.change(screen.getByLabelText('Base environment'), { target: { value: 'base-id' } });
  fireEvent.change(screen.getByLabelText('Dependency policy'), { target: { value: 'adjust' } });
  fireEvent.change(screen.getByLabelText('Conda requirements'), { target: { value: 'numpy >=2,<3\n\nffmpeg >=7' } });
  fireEvent.change(screen.getByLabelText('PyPI requirements'), { target: { value: 'requests>=2,<3' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create revision' }));
  expect(await screen.findByRole('heading', { name: 'my-tools' })).toBeInTheDocument();
  expect(api.createEnvironment).toHaveBeenCalledWith({ name: 'my-tools', baseEnvironmentId: 'base-id', policy: 'adjust', python: '3.12.*', condaDependencies: ['numpy >=2,<3', 'ffmpeg >=7'], pypiDependencies: ['requests>=2,<3'] });
  expect(api.prepareEnvironment).not.toHaveBeenCalled();
  expect(screen.getByText('/data/definitions/my-tools/pixi.toml')).toBeInTheDocument();
  expect(screen.getByText('runtime', { exact: true })).toBeInTheDocument();
});

test('prevents removing referenced or kept revisions', async () => {
  render(<RuntimeEnvironments environments={[{ ...developer, pinned: true }, { ...developer, environmentId: 'used', name: 'used-runtime', extensionIds: ['extension'] }]} extensions={[]} onRefresh={refresh} locked={false} />);
  expect(screen.getByRole('button', { name: 'Remove my-tools' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Remove used-runtime' })).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Keep my-tools' }));
  expect(api.retainEnvironment).toHaveBeenCalledWith('dev-id', false);
  await screen.findByDisplayValue('/data/runtime-storage');
});

test('cancels preparation and exposes actionable errors', async () => {
  api.cancelEnvironmentPreparation.mockRejectedValue(new Error('Cannot stop installer'));
  render(<RuntimeEnvironments environments={[{ ...developer, state: 'preparing' }]} extensions={[]} onRefresh={refresh} locked={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'Cancel preparation' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot stop installer');
  expect(api.cancelEnvironmentPreparation).toHaveBeenCalledWith('dev-id');
});
