import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { ServicesWorkspace } from './ServicesWorkspace';

const api = vi.hoisted(() => ({
  cancelExtensionInstall: vi.fn(), fetchCatalog: vi.fn(), fetchExtensions: vi.fn(), fetchEnvironments: vi.fn(),
  installExtension: vi.fn(), setExtensionEnabled: vi.fn(), uninstallExtension: vi.fn(),
  importExtensionPackage: vi.fn(),
}));
vi.mock('../api/client', () => api);

const vision = { extensionId: 'cvkit', name: 'Computer Vision', version: '1.0.0', description: 'Track objects.',
  state: 'installed', detail: '', serviceClasses: ['f8.cvkit.tracking'], runtimeKind: 'native',
  environmentId: null, preinstalled: true };
const pose = { extensionId: 'mediapipe', name: 'MediaPipe Pose', version: '1.0.0', description: 'Estimate pose.',
  state: 'available', detail: '', serviceClasses: ['f8.mp.pose'], runtimeKind: 'pixi',
  environmentId: null, preinstalled: false };

beforeEach(() => {
  api.fetchCatalog.mockResolvedValue({ services: [{ serviceClass: 'f8.cvkit.tracking', label: 'Tracking' }], operators: [] });
  api.fetchEnvironments.mockResolvedValue([]);
  api.fetchExtensions.mockResolvedValue([vision, pose]);
});
afterEach(() => { cleanup(); vi.resetAllMocks(); });

test('installs a generic extension without dropping the other cards', async () => {
  api.installExtension.mockResolvedValue({ ...pose, state: 'installing', detail: 'Preparing pose runtime' });
  render(<ServicesWorkspace />);
  expect(await screen.findByText('Tracking')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Install MediaPipe Pose' }));
  expect(await screen.findByText('Preparing pose runtime')).toBeInTheDocument();
  expect(screen.getByText('Computer Vision')).toBeInTheDocument();
  expect(api.installExtension).toHaveBeenCalledWith('mediapipe');
});

test('disables an installed native extension', async () => {
  api.fetchExtensions.mockResolvedValueOnce([vision, pose]).mockResolvedValue([{ ...vision, state: 'disabled' }, pose]);
  api.setExtensionEnabled.mockResolvedValue({ ...vision, state: 'disabled' });
  render(<ServicesWorkspace />);
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Enable Computer Vision' }));
  expect(await screen.findByText('disabled')).toBeInTheDocument();
  expect(api.setExtensionEnabled).toHaveBeenCalledWith('cvkit', false);
});

test('uninstalls an extension and offers reinstall', async () => {
  api.fetchExtensions.mockResolvedValueOnce([vision, pose]).mockResolvedValue([{ ...vision, state: 'available' }, pose]);
  api.uninstallExtension.mockResolvedValue({ ...vision, state: 'available' });
  render(<ServicesWorkspace />);
  fireEvent.click(await screen.findByRole('button', { name: 'Uninstall Computer Vision' }));
  expect(await screen.findByRole('button', { name: 'Install Computer Vision' })).toBeInTheDocument();
  expect(api.uninstallExtension).toHaveBeenCalledWith('cvkit');
});

test('shows which extensions share a runtime', async () => {
  api.fetchEnvironments.mockResolvedValue([{ environmentId: 'base', runtimeKind: 'bundled',
    ready: true, extensionIds: ['cvkit', 'mediapipe'] }]);
  render(<ServicesWorkspace />);
  const runtimes = await screen.findByLabelText('Shared runtimes');
  expect(within(runtimes).getByText('Computer Vision, MediaPipe Pose · ready')).toBeInTheDocument();
});

test('keeps the extension installed and reports a rejected uninstall', async () => {
  api.uninstallExtension.mockRejectedValue(new Error('Stop running services before uninstalling the extension'));
  render(<ServicesWorkspace />);
  fireEvent.click(await screen.findByRole('button', { name: 'Uninstall Computer Vision' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Stop running services');
  expect(screen.getByRole('checkbox', { name: 'Enable Computer Vision' })).toBeChecked();
});

test('imports a package from a publisher and adds its extension card', async () => {
  api.importExtensionPackage.mockResolvedValue([vision, pose, { ...vision, extensionId: 'player', name: 'Player' }]);
  render(<ServicesWorkspace />);
  await screen.findByText('Computer Vision');
  fireEvent.change(screen.getByLabelText('Extension package URL'), { target: { value: 'https://publisher.example/player.zip' } });
  fireEvent.change(screen.getByLabelText('Extension package SHA-256'), { target: { value: 'a'.repeat(64) } });
  fireEvent.click(screen.getByRole('button', { name: 'Add package' }));
  expect(await screen.findByText('Player')).toBeInTheDocument();
  expect(api.importExtensionPackage).toHaveBeenCalledWith('https://publisher.example/player.zip', 'a'.repeat(64));
});

test('restores the toggle if a running service prevents disabling', async () => {
  api.setExtensionEnabled.mockRejectedValue(new Error('Stop running services before disabling the extension'));
  render(<ServicesWorkspace />);
  const toggle = await screen.findByRole('checkbox', { name: 'Enable Computer Vision' });
  fireEvent.click(toggle);
  expect(toggle).not.toBeChecked();
  expect(await screen.findByRole('alert')).toHaveTextContent('Stop running services');
  expect(toggle).toBeChecked();
});

test('explains shared-runtime installation without an environment download', async () => {
  api.fetchExtensions.mockResolvedValue([{ ...pose, runtimeKind: 'shared' }]);
  api.installExtension.mockResolvedValue({ ...pose, runtimeKind: 'shared', state: 'installing' });
  render(<ServicesWorkspace />);
  expect(await screen.findByText('Reuses an installed official environment. No additional environment download.')).toBeInTheDocument();
  expect(screen.queryByText('Runtime dependencies may need to be downloaded. Shared runtimes are reused.')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Install MediaPipe Pose' }));
  expect(await screen.findByText('installing')).toBeInTheDocument();
  expect(api.installExtension).toHaveBeenCalledWith('mediapipe');
});

test('polls extension progress and refreshes the catalog only after installation finishes', async () => {
  vi.useFakeTimers();
  try {
    api.fetchExtensions.mockResolvedValueOnce([vision, { ...pose, state: 'installing' }])
      .mockResolvedValueOnce([vision, { ...pose, state: 'installing', detail: 'Checking dependencies' }])
      .mockResolvedValue([vision, { ...pose, state: 'installed' }]);
    render(<ServicesWorkspace />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('installing')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByText('Checking dependencies')).toBeInTheDocument();
    expect(api.fetchExtensions).toHaveBeenCalledTimes(2);
    expect(api.fetchCatalog).toHaveBeenCalledTimes(1);
    expect(api.fetchEnvironments).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.queryByText('installing')).not.toBeInTheDocument();
    expect(api.fetchCatalog).toHaveBeenCalledTimes(2);
    expect(api.fetchEnvironments).toHaveBeenCalledTimes(2);
  } finally {
    vi.useRealTimers();
  }
});
