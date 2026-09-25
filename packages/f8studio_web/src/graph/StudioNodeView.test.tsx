import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import type { ServiceNode, StateSpec } from '../api/contracts';
import { GraphNodeInteractionContext, StudioNodeView } from './StudioNodeView';
import type { StudioFlowNode } from './projection';

vi.mock('./useRuntimeNodeState', () => ({
  useRuntimeNodeState: () => ({
    availableDevices: {
      field: 'availableDevices', found: true,
      value: ['Auto', 'Recording: USB Mic'], tsMs: 1,
    },
  }),
}));

afterEach(cleanup);

test('shows a live device selector on a write-only service state port', () => {
  const selectedDevice: StateSpec = {
    name: 'selectedDevice', label: 'Capture Device', access: 'wo', showOnNode: true,
    control: { kind: 'select', optionsFromState: 'availableDevices' }, valueSchema: { type: 'string', default: 'Auto' },
  };
  const node: ServiceNode = {
    kind: 'service', nodeId: 'capture', name: 'Audio Capture', serviceId: 'capture',
    serviceClass: 'f8.audiocap', enabled: true, stateValues: {},
    spec: {
      serviceClass: 'f8.audiocap', label: 'Audio Capture', specKind: 'service',
      stateFields: [selectedDevice],
    },
    ports: [{
      portId: 'state:input:selectedDevice', name: 'selectedDevice', runtimeName: 'selectedDevice',
      kind: 'state', direction: 'input', stateSpec: selectedDevice,
    }],
  };
  const setState = vi.fn();
  const interaction = {
    busy: false,
    pendingCommands: new Set<string>(),
    connectedStateInputs: new Set<string>(),
    resizeService: vi.fn(),
    setState,
    openCommand: vi.fn(),
    showOutput: vi.fn(),
  };
  const props = { id: 'capture', data: { graphNode: node, childCount: 0 }, selected: false } as NodeProps<StudioFlowNode>;

  render(<ReactFlowProvider>
    <GraphNodeInteractionContext.Provider value={interaction}>
      <StudioNodeView {...props} />
    </GraphNodeInteractionContext.Provider>
  </ReactFlowProvider>);

  const select = screen.getByRole('combobox', { name: 'Capture Device' });
  expect(screen.getByText('Capture Device')).toBeInTheDocument();
  expect(select).toHaveValue('"Auto"');
  fireEvent.change(select, { target: { value: '"Recording: USB Mic"' } });
  expect(setState).toHaveBeenCalledWith('capture', 'selectedDevice', 'Recording: USB Mic');
});
