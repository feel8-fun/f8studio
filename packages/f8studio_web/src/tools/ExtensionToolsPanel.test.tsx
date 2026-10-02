import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExtensionToolsPanel } from './ExtensionToolsPanel';
import { cancelToolJob, fetchExtensions, fetchExtensionTools, fetchToolJobs, runExtensionTool } from '../api/client';

vi.mock('../api/client', () => ({ fetchExtensions: vi.fn(), fetchExtensionTools: vi.fn(), fetchToolJobs: vi.fn(), runExtensionTool: vi.fn(), cancelToolJob: vi.fn() }));

afterEach(cleanup);

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(fetchExtensions).mockResolvedValue([]);
  vi.mocked(fetchToolJobs).mockResolvedValue([]);
});

describe('ExtensionToolsPanel', () => {
  it('explains how to obtain tools when no extension provides them', async () => {
    vi.mocked(fetchExtensionTools).mockResolvedValue([]);
    render(<ExtensionToolsPanel />);
    await waitFor(() => expect(fetchExtensionTools).toHaveBeenCalled());
    expect(screen.getByText('No tools are installed and enabled.')).toBeInTheDocument();
  });

  it('requires confirmation, submits typed arguments, and displays a failed result', async () => {
    vi.mocked(fetchExtensionTools).mockResolvedValue([{ extensionId: 'example', toolId: 'inspect', name: 'Inspect', description: 'Inspect a target', requiresConfirmation: true, allowConcurrent: false,
      fields: [{ name: 'target', label: 'Target', kind: 'string', required: true, default: null, choices: [] }, { name: 'port', label: 'Port', kind: 'integer', default: 39540, required: false, choices: [] }] }]);
    vi.mocked(runExtensionTool).mockResolvedValue({ jobId: 'job', extensionId: 'example', extensionVersion: '1.0', toolId: 'inspect', arguments: {}, status: 'failed', createdAt: '2026-10-01', updatedAt: '2026-10-01', result: null, error: 'Target unavailable', log: 'Diagnostic detail' });
    render(<ExtensionToolsPanel />);
    fireEvent.click(await screen.findByRole('button', { name: 'Inspect' }));
    fireEvent.change(screen.getByLabelText('Target'), { target: { value: '/games/example' } });
    expect(screen.getByRole('button', { name: 'Run tool' })).toBeDisabled();
    fireEvent.click(screen.getByLabelText('I confirm execution of this tool with these inputs.'));
    fireEvent.click(screen.getByRole('button', { name: 'Run tool' }));
    await waitFor(() => expect(runExtensionTool).toHaveBeenCalledWith('example', 'inspect', { target: '/games/example', port: 39540 }, true));
    expect(await screen.findByText('Target unavailable')).toBeInTheDocument();
    expect(screen.getByText('Diagnostic detail')).toBeInTheDocument();
  });
});

it('keeps a stream running while another tool executes, and offers Stop after remounting', async () => {
  const stream = { extensionId: 'diagnostics', toolId: 'send', name: 'Send stream', description: 'Send until stopped', requiresConfirmation: false, allowConcurrent: true, fields: [] };
  const verify = { ...stream, toolId: 'verify', name: 'Verify stream' };
  const active = { jobId: 'stream-job', extensionId: 'diagnostics', extensionVersion: '0.2.0', toolId: 'send', arguments: {}, status: 'running' as const, createdAt: '2026-10-02', updatedAt: '2026-10-02', result: null, error: '', log: '' };
  vi.mocked(fetchExtensionTools).mockResolvedValue([stream, verify]);
  vi.mocked(fetchToolJobs).mockResolvedValue([active]);
  vi.mocked(cancelToolJob).mockResolvedValue({ ...active, status: 'cancelled' });
  const first = render(<ExtensionToolsPanel />);
  fireEvent.click(await screen.findByRole('button', { name: 'Verify stream' }));
  expect(screen.getByRole('button', { name: 'Run tool' })).toBeEnabled();
  first.unmount();
  expect(cancelToolJob).not.toHaveBeenCalled();
  render(<ExtensionToolsPanel />);
  await screen.findByRole('button', { name: 'Task running' });
  fireEvent.click(screen.getAllByRole('button', { name: 'Stop' })[0]!);
  await waitFor(() => expect(cancelToolJob).toHaveBeenCalledWith('stream-job'));
  expect(await screen.findByRole('button', { name: 'Run tool' })).toBeEnabled();
});
