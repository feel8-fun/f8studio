import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ToolsWorkspace } from './ToolsWorkspace';
import { fetchExtensionTools, fetchToolJobs, runExtensionTool } from '../api/client';

vi.mock('../api/client', () => ({ fetchExtensionTools: vi.fn(), fetchToolJobs: vi.fn(), runExtensionTool: vi.fn(), cancelToolJob: vi.fn() }));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(fetchToolJobs).mockResolvedValue([]);
});

describe('ToolsWorkspace', () => {
  it('explains how to obtain tools when no extension provides them', async () => {
    vi.mocked(fetchExtensionTools).mockResolvedValue([]);
    render(<ToolsWorkspace />);
    await waitFor(() => expect(fetchExtensionTools).toHaveBeenCalled());
    expect(screen.getByText('No tools are installed and enabled.')).toBeInTheDocument();
  });

  it('requires confirmation, submits typed arguments, and displays a failed result', async () => {
    vi.mocked(fetchExtensionTools).mockResolvedValue([{ extensionId: 'example', toolId: 'inspect', name: 'Inspect', description: 'Inspect a target', requiresConfirmation: true,
      fields: [{ name: 'target', label: 'Target', kind: 'string', required: true, default: null, choices: [] }, { name: 'port', label: 'Port', kind: 'integer', default: 39540, required: false, choices: [] }] }]);
    vi.mocked(runExtensionTool).mockResolvedValue({ jobId: 'job', extensionId: 'example', extensionVersion: '1.0', toolId: 'inspect', arguments: {}, status: 'failed', createdAt: '2026-10-01', updatedAt: '2026-10-01', result: null, error: 'Target unavailable', log: 'Diagnostic detail' });
    render(<ToolsWorkspace />);
    await screen.findByRole('option', { name: 'Inspect (example)' });
    fireEvent.change(screen.getByLabelText('Tool'), { target: { value: 'example/inspect' } });
    fireEvent.change(screen.getByLabelText('Target'), { target: { value: '/games/example' } });
    expect(screen.getByRole('button', { name: 'Run tool' })).toBeDisabled();
    fireEvent.click(screen.getByLabelText('I confirm execution of this tool with these inputs.'));
    fireEvent.click(screen.getByRole('button', { name: 'Run tool' }));
    await waitFor(() => expect(runExtensionTool).toHaveBeenCalledWith('example', 'inspect', { target: '/games/example', port: 39540 }, true));
    expect(await screen.findByText('Target unavailable')).toBeInTheDocument();
    expect(screen.getByText('Diagnostic detail')).toBeInTheDocument();
  });
});
