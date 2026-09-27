import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import type { OperatorNode, ServiceNode, StateSpec } from '../api/contracts';
import { GraphNodeInteractionContext, StudioNodeView } from './StudioNodeView';
import type { StudioFlowNode } from './projection';

const presentationOutput = vi.hoisted(() => vi.fn());
const emulatorWrite = vi.hoisted(() => vi.fn());
vi.mock('../presentation/PresentationStore', () => ({ usePresentationOutput: presentationOutput }));
vi.mock('osr-emu', () => ({ OSREmulator: class { write = emulatorWrite; destroy() {} } }));
vi.mock('../presentation/PresentationWave', () => ({
  PresentationWave: ({ payload }: { payload: { series: unknown } }) => <div data-testid="wave-content">{JSON.stringify(payload.series)}</div>,
}));

vi.mock('./useRuntimeNodeState', () => ({
  useRuntimeNodeState: () => ({
    availableDevices: {
      field: 'availableDevices', found: true,
      value: ['Auto', 'Recording: USB Mic'], tsMs: 1,
    },
  }),
}));

afterEach(cleanup);

test('shows wave data in the node and holds the last frame when UI updates are paused', () => {
  const node: OperatorNode = {
    kind: 'operator', nodeId: 'wave', name: 'Wave Viz', serviceId: 'studio', serviceClass: 'f8.pystudio',
    operatorClass: 'f8.viz.wave', enabled: true, stateValues: { uiUpdate: true }, ports: [],
    spec: { serviceClass: 'f8.pystudio', operatorClass: 'f8.viz.wave', label: 'Wave Viz',
      specKind: 'operator', rendererClass: 'viz_wave', stateFields: [] },
  };
  presentationOutput.mockReturnValue({ renderer: 'wave', payload: { series: { x: [[1, 2]] } } });
  const props = { id: 'wave', data: { graphNode: node, childCount: 0 }, selected: false } as NodeProps<StudioFlowNode>;
  const view = render(<ReactFlowProvider><StudioNodeView {...props} /></ReactFlowProvider>);
  expect(screen.getByTestId('wave-content')).toHaveTextContent('[[1,2]]');

  presentationOutput.mockReturnValue({ renderer: 'wave', payload: { series: { x: [[2, 3]] } } });
  view.rerender(<ReactFlowProvider><StudioNodeView {...props} data={{ ...props.data, graphNode: {
    ...node, stateValues: { uiUpdate: false },
  } }} /></ReactFlowProvider>);
  expect(screen.getByTestId('wave-content')).toHaveTextContent('[[1,2]]');
});

test('shows a 3D TCode preview and forwards its command to the emulator', async () => {
  const node: OperatorNode = {
    kind: 'operator', nodeId: 'tcode', name: 'TCode Viz', serviceId: 'studio', serviceClass: 'f8.pystudio',
    operatorClass: 'f8.viz.tcode', enabled: true, stateValues: {}, ports: [],
    spec: { serviceClass: 'f8.pystudio', operatorClass: 'f8.viz.tcode', label: 'TCode Viz',
      specKind: 'operator', rendererClass: 'viz_tcode', stateFields: [] },
  };
  presentationOutput.mockReturnValue({ renderer: 'tcode', payload: { model: 'SR6', line: 'L05000 R09999\n' } });
  const props = { id: 'tcode', data: { graphNode: node, childCount: 0 }, selected: false } as NodeProps<StudioFlowNode>;
  render(<ReactFlowProvider><StudioNodeView {...props} /></ReactFlowProvider>);

  await waitFor(() => expect(screen.getByLabelText('SR6 3D TCode visualizer')).toBeInTheDocument());
  expect(emulatorWrite).toHaveBeenCalledWith('L05000 R09999\n');
});

test('allows switching an undeployed TCode node to channel bars', async () => {
  const node: OperatorNode = {
    kind: 'operator', nodeId: 'tcode-empty-preview', name: 'TCode Viz', serviceId: 'studio', serviceClass: 'f8.pystudio',
    operatorClass: 'f8.viz.tcode', enabled: true, stateValues: { model: 'SSR1' }, ports: [],
    spec: { serviceClass: 'f8.pystudio', operatorClass: 'f8.viz.tcode', label: 'TCode Viz',
      specKind: 'operator', rendererClass: 'viz_tcode', stateFields: [] },
  };
  presentationOutput.mockReturnValue(undefined);
  const props = { id: node.nodeId, data: { graphNode: node, childCount: 0 }, selected: false } as NodeProps<StudioFlowNode>;
  render(<ReactFlowProvider><StudioNodeView {...props} /></ReactFlowProvider>);

  const barsButton = await screen.findByRole('button', { name: 'Show channel bars' });
  fireEvent.click(barsButton);
  expect(barsButton).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByTestId('tcode-preview-tcode-empty-preview')).toHaveTextContent('Waiting for TCode');
  expect(screen.queryByLabelText('SSR1 3D TCode visualizer')).not.toBeInTheDocument();
});

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
