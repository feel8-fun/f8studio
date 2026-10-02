import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { ExtensionsWorkspace } from './ExtensionsWorkspace';

const api = vi.hoisted(() => ({
  fetchExtensionDetail: vi.fn(), cancelExtensionInstall: vi.fn(), fetchExtensions: vi.fn(), fetchEnvironments: vi.fn(),
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
  window.history.replaceState(null, "", "/?view=extensions");
  api.fetchEnvironments.mockResolvedValue([]);
  api.fetchExtensions.mockResolvedValue([vision, pose]);
});
afterEach(() => { cleanup(); vi.resetAllMocks(); });

test('installs a generic extension without dropping the other cards', async () => {
  api.installExtension.mockResolvedValue({ ...pose, state: 'installing', detail: 'Preparing pose runtime' });
  render(<ExtensionsWorkspace />);
  expect(await screen.findByText('MediaPipe Pose')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Install MediaPipe Pose' }));
  expect(await screen.findByText('Preparing pose runtime')).toBeInTheDocument();
  expect(screen.getByText('Computer Vision')).toBeInTheDocument();
  expect(api.installExtension).toHaveBeenCalledWith('mediapipe');
});

test('disables an installed native extension', async () => {
  api.fetchExtensions.mockResolvedValueOnce([vision, pose]).mockResolvedValue([{ ...vision, state: 'disabled' }, pose]);
  api.setExtensionEnabled.mockResolvedValue({ ...vision, state: 'disabled' });
  render(<ExtensionsWorkspace />);
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Enable Computer Vision' }));
  expect(await screen.findByText('disabled')).toBeInTheDocument();
  expect(api.setExtensionEnabled).toHaveBeenCalledWith('cvkit', false);
});

test('uninstalls an extension and offers reinstall', async () => {
  api.fetchExtensions.mockResolvedValueOnce([vision, pose]).mockResolvedValue([{ ...vision, state: 'available' }, pose]);
  api.uninstallExtension.mockResolvedValue({ ...vision, state: 'available' });
  render(<ExtensionsWorkspace />);
  fireEvent.click(await screen.findByRole('button', { name: 'Uninstall Computer Vision' }));
  expect(await screen.findByRole('button', { name: 'Install Computer Vision' })).toBeInTheDocument();
  expect(api.uninstallExtension).toHaveBeenCalledWith('cvkit');
});

test('keeps runtime management on its separate workspace', async () => {
  render(<ExtensionsWorkspace />);
  await screen.findByText('Computer Vision');
  expect(screen.queryByRole('region', { name: 'Runtime environments' })).not.toBeInTheDocument();
});

test('keeps the extension installed and reports a rejected uninstall', async () => {
  api.uninstallExtension.mockRejectedValue(new Error('Stop running services before uninstalling the extension'));
  render(<ExtensionsWorkspace />);
  fireEvent.click(await screen.findByRole('button', { name: 'Uninstall Computer Vision' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Stop running services');
  expect(screen.getByRole('checkbox', { name: 'Enable Computer Vision' })).toBeChecked();
});

test('imports a package from a publisher and adds its extension card', async () => {
  api.importExtensionPackage.mockResolvedValue([vision, pose, { ...vision, extensionId: 'player', name: 'Player' }]);
  render(<ExtensionsWorkspace />);
  await screen.findByText('Computer Vision');
  fireEvent.change(screen.getByLabelText('Extension package URL'), { target: { value: 'https://publisher.example/player.zip' } });
  fireEvent.change(screen.getByLabelText('Extension package SHA-256'), { target: { value: 'a'.repeat(64) } });
  fireEvent.click(screen.getByRole('button', { name: 'Add package' }));
  expect(await screen.findByText('Player')).toBeInTheDocument();
  expect(api.importExtensionPackage).toHaveBeenCalledWith('https://publisher.example/player.zip', 'a'.repeat(64));
});

test('restores the toggle if a running service prevents disabling', async () => {
  api.setExtensionEnabled.mockRejectedValue(new Error('Stop running services before disabling the extension'));
  render(<ExtensionsWorkspace />);
  const toggle = await screen.findByRole('checkbox', { name: 'Enable Computer Vision' });
  fireEvent.click(toggle);
  expect(toggle).not.toBeChecked();
  expect(await screen.findByRole('alert')).toHaveTextContent('Stop running services');
  expect(toggle).toBeChecked();
});

test('explains shared-runtime installation without an environment download', async () => {
  api.fetchExtensions.mockResolvedValue([{ ...pose, runtimeKind: 'shared' }]);
  api.installExtension.mockResolvedValue({ ...pose, runtimeKind: 'shared', state: 'installing' });
  render(<ExtensionsWorkspace />);
  expect(await screen.findByText('Reuses an installed official environment. No additional environment download.')).toBeInTheDocument();
  expect(screen.queryByText('Runtime dependencies may need to be downloaded. Shared runtimes are reused.')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Install MediaPipe Pose' }));
  expect(await screen.findByText('installing')).toBeInTheDocument();
  expect(api.installExtension).toHaveBeenCalledWith('mediapipe');
});

test('polls extension progress and refreshes runtimes only after installation finishes', async () => {
  vi.useFakeTimers();
  try {
    api.fetchExtensions.mockResolvedValueOnce([vision, { ...pose, state: 'installing' }])
      .mockResolvedValueOnce([vision, { ...pose, state: 'installing', detail: 'Checking dependencies' }])
      .mockResolvedValue([vision, { ...pose, state: 'installed' }]);
    render(<ExtensionsWorkspace />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('installing')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByText('Checking dependencies')).toBeInTheDocument();
    expect(api.fetchExtensions).toHaveBeenCalledTimes(2);
    expect(api.fetchEnvironments).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.queryByText('installing')).not.toBeInTheDocument();
    expect(api.fetchEnvironments).toHaveBeenCalledTimes(2);
  } finally {
    vi.useRealTimers();
  }
});

test('searches extensions while keeping package installation available', async () => {
  render(<ExtensionsWorkspace />);
  await screen.findByText('Computer Vision');
  expect(screen.queryByRole('region', { name: 'Enabled services' })).not.toBeInTheDocument();
  expect(screen.getByText('2 extensions')).toBeInTheDocument();
  fireEvent.change(screen.getByRole('textbox', { name: 'Search extensions' }), { target: { value: 'mediapipe' } });
  expect(screen.getByText('MediaPipe Pose')).toBeInTheDocument();
  expect(screen.queryByText('Computer Vision')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Extension package URL')).toBeInTheDocument();
  fireEvent.change(screen.getByRole('textbox', { name: 'Search extensions' }), { target: { value: 'missing' } });
  expect(screen.getByText('No matching extensions')).toBeInTheDocument();
});

const visionDetail = { extensionId: 'cvkit', services: [{ serviceClass: 'f8.cvkit.tracking', describe: {
  service: { serviceClass: 'f8.cvkit.tracking', label: 'Tracking', description: 'Track image targets.',
    stateFields: [{ name: 'threshold', valueSchema: { type: 'number' }, access: 'rw', description: 'Detection threshold' }],
    dataInPorts: [], dataOutPorts: [], commands: [] }, operators: [],
} }], tools: [{ toolId: 'inspect', name: 'Inspect target', description: 'Inspect a selected target.',
  timeoutSeconds: null, requiresConfirmation: true, allowConcurrent: true,
  fields: [{ name: 'target', label: 'Target path', kind: 'string', required: true }] }],
  skills: [{ skillId: 'workflow', content: '# Workflow\nInspect the target before running the tool.' }] };

test('opens extension contents and service, tool and skill details with package controls', async () => {
  api.fetchExtensionDetail.mockResolvedValue(visionDetail);
  render(<ExtensionsWorkspace />);
  fireEvent.click(await screen.findByRole('link', { name: 'Computer Vision' }));
  expect(await screen.findByRole('link', { name: /Tracking/ })).toBeInTheDocument();
  expect(screen.getByRole('checkbox', { name: 'Enable Computer Vision' })).toBeChecked();
  expect(window.location.search).toContain('extension=cvkit');
  fireEvent.click(screen.getByRole('link', { name: /Tracking/ }));
  const service = await screen.findByRole('article', { name: 'service details' });
  expect(within(service).getByText('Detection threshold')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Uninstall Computer Vision' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('link', { name: 'Back to extension contents' }));
  fireEvent.click(screen.getByRole('link', { name: /Inspect target/ }));
  const tool = await screen.findByRole('article', { name: 'tool details' });
  expect(within(tool).getByText('Runs until stopped')).toBeInTheDocument();
  expect(within(tool).getByText('Target path')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('link', { name: 'Back to extension contents' }));
  fireEvent.click(screen.getByRole('link', { name: /workflow/ }));
  expect(await screen.findByText(/Inspect the target before/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('link', { name: 'All extensions' }));
  expect(screen.getByRole('textbox', { name: 'Search extensions' })).toBeInTheDocument();
  expect(api.fetchExtensionDetail).toHaveBeenCalledTimes(1);
});

test('supports direct detail URLs and enabling a disabled extension from skill details', async () => {
  window.history.replaceState(null, '', '/?view=extensions&extension=cvkit&skill=workflow');
  api.fetchExtensionDetail.mockResolvedValue(visionDetail);
  api.fetchExtensions.mockResolvedValueOnce([{ ...vision, state: 'disabled' }, pose]).mockResolvedValue([vision, pose]);
  api.setExtensionEnabled.mockResolvedValue(vision);
  render(<ExtensionsWorkspace />);
  expect(await screen.findByText(/Inspect the target before/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Computer Vision' }));
  expect(await screen.findByText('installed')).toBeInTheDocument();
  expect(api.setExtensionEnabled).toHaveBeenCalledWith('cvkit', true);
  expect(screen.getByRole('article', { name: 'skill details' })).toBeInTheDocument();
});

test('previews an available extension and installs it from its detail page', async () => {
  api.fetchExtensionDetail.mockResolvedValue({ extensionId: 'mediapipe', services: [], tools: [], skills: [] });
  api.installExtension.mockResolvedValue({ ...pose, state: 'installing' });
  render(<ExtensionsWorkspace />);
  fireEvent.click(await screen.findByRole('link', { name: 'MediaPipe Pose' }));
  expect(await screen.findByText('No tools declared.')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Install MediaPipe Pose' }));
  expect(await screen.findByText('installing')).toBeInTheDocument();
  expect(api.installExtension).toHaveBeenCalledWith('mediapipe');
});

test('reports detail errors and retries without losing extension actions', async () => {
  api.fetchExtensionDetail.mockRejectedValueOnce(new Error('Cannot read package metadata')).mockResolvedValue(visionDetail);
  render(<ExtensionsWorkspace />);
  fireEvent.click(await screen.findByRole('link', { name: 'Computer Vision' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot read package metadata');
  expect(screen.getByRole('checkbox', { name: 'Enable Computer Vision' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Retry details' }));
  expect(await screen.findByRole('link', { name: /Tracking/ })).toBeInTheDocument();
});


test('tracks environment preparation in extension details and restores controls after it finishes', async () => {
  vi.useFakeTimers();
  try {
    window.history.replaceState(null, '', '/?view=extensions&extension=cvkit');
    api.fetchExtensionDetail.mockResolvedValue(visionDetail);
    api.fetchEnvironments.mockResolvedValueOnce([{ environmentId: 'base', runtimeKind: 'workspace', ready: false, extensionIds: ['cvkit'], state: 'preparing' }])
      .mockResolvedValue([{ environmentId: 'base', runtimeKind: 'workspace', ready: true, extensionIds: ['cvkit'], state: 'ready' }]);
    render(<ExtensionsWorkspace />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByRole('checkbox', { name: 'Enable Computer Vision' })).toBeDisabled();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByRole('checkbox', { name: 'Enable Computer Vision' })).toBeEnabled();
    expect(api.fetchEnvironments).toHaveBeenCalledTimes(2);
  } finally { vi.useRealTimers(); }
});
