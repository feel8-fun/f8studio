import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ToolsWorkspace } from './ToolsWorkspace';
import { fetchExtensions, fetchExtensionTools, fetchToolJobs } from '../api/client';

vi.mock('../api/client', () => ({
  fetchExtensions: vi.fn(), fetchExtensionTools: vi.fn(), fetchToolJobs: vi.fn(), runExtensionTool: vi.fn(), cancelToolJob: vi.fn(),
}));
afterEach(cleanup);
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(fetchExtensions).mockResolvedValue([]);
  vi.mocked(fetchExtensionTools).mockResolvedValue([]);
  vi.mocked(fetchToolJobs).mockResolvedValue([]);
});
it('has no built-in diagnostics or automation tools', async () => {
  render(<ToolsWorkspace />);
  expect(await screen.findByText('No tools are installed and enabled.')).toBeInTheDocument();
  for (const label of ['Serial devices', 'Skeleton UDP', 'Global hotkeys', 'Managed processes']) {
    expect(screen.queryByRole('button', { name: label })).not.toBeInTheDocument();
  }
});

it('renders game tools only from extension declarations using their fields', async () => {
  vi.mocked(fetchExtensionTools).mockResolvedValue([{
    extensionId: 'example.game-tools', toolId: 'inspect', name: 'Game profile inspector',
    description: 'Description supplied by the extension', requiresConfirmation: false, allowConcurrent: false,
    fields: [{ name: 'directory', label: 'Extension target directory', kind: 'string', required: true, default: '/games', choices: [] }],
  }]);
  vi.mocked(fetchToolJobs).mockResolvedValue([]);
  render(<ToolsWorkspace />);
  fireEvent.click(await screen.findByRole('button', { name: 'Game profile inspector' }));
  expect(screen.getByLabelText('Extension target directory')).toHaveValue('/games');
  expect(screen.getByText('Description supplied by the extension')).toBeInTheDocument();
  expect(screen.queryByText('Executable or game directory')).not.toBeInTheDocument();
});

it('groups tools by extension and selects another tool when the current extension disappears', async () => {
  const inspect = { extensionId: 'debug', toolId: 'inspect', name: 'Inspect stream', description: 'Inspect', requiresConfirmation: false, allowConcurrent: false,
    fields: [{ name: 'port', label: 'Port', kind: 'integer' as const, required: true, default: 39540, choices: [] }] };
  const simulate = { ...inspect, toolId: 'simulate', name: 'Simulate stream' };
  const exportTool = { ...inspect, extensionId: 'export', toolId: 'export', name: 'Export assets' };
  vi.mocked(fetchExtensionTools).mockResolvedValue([inspect, simulate, exportTool]);
  vi.mocked(fetchExtensions).mockResolvedValue([
    { extensionId: 'debug', name: 'Debugging toolkit', version: '1.0', description: '', state: 'installed', detail: '', serviceClasses: [], runtimeEnvironment: null, runtimeSelectable: false, runtimeKind: 'workspace', environmentId: null, preinstalled: false, toolIds: ['inspect', 'simulate'], skillIds: [], resourceIds: [] },
    { extensionId: 'export', name: 'Asset toolkit', version: '1.0', description: '', state: 'installed', detail: '', serviceClasses: [], runtimeEnvironment: null, runtimeSelectable: false, runtimeKind: 'workspace', environmentId: null, preinstalled: false, toolIds: ['export'], skillIds: [], resourceIds: [] },
  ]);
  render(<ToolsWorkspace />);
  const debugging = await screen.findByRole('region', { name: 'Debugging toolkit' });
  expect(within(debugging).getByRole('button', { name: 'Inspect stream' })).toBeInTheDocument();
  expect(within(debugging).getByRole('button', { name: 'Simulate stream' })).toBeInTheDocument();
  expect(within(debugging).queryByRole('button', { name: 'Export assets' })).not.toBeInTheDocument();
  expect(within(screen.getByRole('region', { name: 'Asset toolkit' })).getByRole('button', { name: 'Export assets' })).toBeInTheDocument();
  expect(await screen.findByLabelText('Port')).toHaveValue(39540);
  fireEvent.change(screen.getByLabelText('Port'), { target: { value: '40000' } });
  fireEvent.click(screen.getByRole('button', { name: 'Simulate stream' }));
  expect(screen.getByLabelText('Port')).toHaveValue(39540);
  vi.mocked(fetchExtensionTools).mockResolvedValue([exportTool]);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh tools' }));
  await waitFor(() => expect(screen.queryByRole('region', { name: 'Debugging toolkit' })).not.toBeInTheDocument());
  expect(await screen.findByRole('heading', { name: 'Export assets' })).toBeInTheDocument();
});
