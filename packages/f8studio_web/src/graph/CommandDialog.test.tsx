import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { invokeRuntimeCommand, setRuntimeState } from '../api/client';
import type { CommandSpec, GraphNode } from '../api/contracts';
import { CommandDialog } from './CommandDialog';

vi.mock('../api/client', () => ({
  invokeRuntimeCommand: vi.fn(),
  setRuntimeState: vi.fn(),
}));

const command: CommandSpec = {
  name: 'Run',
  params: [
    { name: 'count', valueSchema: { type: 'integer', minimum: 1 }, valueRequired: true },
    { name: 'label', valueSchema: { type: 'string' } },
  ],
};

const service: GraphNode = {
  kind: 'service', nodeId: 'svc-1', serviceId: 'svc-1', serviceClass: 'test.service',
  name: 'Service', enabled: true, ports: [], stateValues: {},
  spec: { specKind: 'service', serviceClass: 'test.service', label: 'Service', commands: [command] },
};

const operator: GraphNode = {
  kind: 'operator', nodeId: 'op-1', serviceId: 'svc-1', serviceClass: 'test.service',
  operatorClass: 'test.operator', name: 'Operator', enabled: true, stateValues: {},
  ports: [{ portId: 'command:input:Run', name: 'Run', runtimeName: '__cmd__.run.in', kind: 'command', direction: 'input' }],
  spec: { specKind: 'operator', serviceClass: 'test.service', operatorClass: 'test.operator', label: 'Operator', commands: [command] },
};

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(cleanup);

test('validates parameters and invokes a service command with typed values', async () => {
  vi.mocked(invokeRuntimeCommand).mockResolvedValue({ success: true, result: 3 });
  const onClose = vi.fn();
  const onResult = vi.fn();
  render(<CommandDialog node={service} command={command} onClose={onClose} onResult={onResult} />);
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('count is required');
  expect(invokeRuntimeCommand).not.toHaveBeenCalled();

  fireEvent.change(screen.getByRole('spinbutton', { name: 'count *' }), { target: { value: '3' } });
  fireEvent.change(screen.getByRole('textbox', { name: 'label' }), { target: { value: 'sample' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  await waitFor(() => expect(invokeRuntimeCommand).toHaveBeenCalledWith('svc-1', 'Run', { count: 3, label: 'sample' }));
  await waitFor(() => expect(onResult).toHaveBeenCalledWith('success', 'Service: Run', '3'));
  expect(onClose).toHaveBeenCalledOnce();
});

test('submits an operator command through its declared command input', async () => {
  vi.mocked(setRuntimeState).mockResolvedValue({ success: true });
  const onResult = vi.fn();
  render(<CommandDialog node={operator} command={command} onClose={() => undefined} onResult={onResult} />);
  fireEvent.change(screen.getByRole('spinbutton', { name: 'count *' }), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  await waitFor(() => expect(setRuntimeState).toHaveBeenCalledWith('svc-1', 'op-1', '__cmd__.run.in', { count: 2 }));
  await waitFor(() => expect(onResult).toHaveBeenCalledWith('success', 'Operator: Run', 'Submitted to runtime'));
});

test('reports runtime rejection and keeps the parameter dialog open', async () => {
  vi.mocked(invokeRuntimeCommand).mockResolvedValue({ success: false, errorMessage: 'Player is offline' });
  const onClose = vi.fn();
  const onResult = vi.fn();
  render(<CommandDialog node={service} command={command} onClose={onClose} onResult={onResult} />);
  fireEvent.change(screen.getByRole('spinbutton', { name: 'count *' }), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Player is offline');
  expect(onResult).toHaveBeenCalledWith('error', 'Service: Run failed', 'Player is offline');
  expect(onClose).not.toHaveBeenCalled();
});
