import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import { TCodeView } from './TCodeView';
import { setTCodeViewMode } from './tcodeViewMode';

const emulator = vi.hoisted(() => ({ create: vi.fn(), write: vi.fn(), destroy: vi.fn() }));
vi.mock('osr-emu', () => ({
  OSREmulator: class {
    constructor(target: HTMLElement, options: { model: string }) { emulator.create(target, options); }
    write(line: string) { emulator.write(line); }
    destroy() { emulator.destroy(); }
  },
}));

afterEach(() => { cleanup(); vi.clearAllMocks(); });

test('replays commands to the selected 3D model and rebuilds on reset', async () => {
  const view = render(<TCodeView nodeId="tcode-unit-3d" payload={{ model: 'SR6', line: 'L05000I500' }} />);
  await waitFor(() => expect(emulator.create).toHaveBeenCalledWith(expect.any(HTMLElement), { model: 'SR6' }));
  expect(emulator.write).toHaveBeenCalledWith('L05000I500\n');
  expect(screen.getByText('5000')).toBeInTheDocument();
  expect(screen.queryByText('I5')).not.toBeInTheDocument();

  view.rerender(<TCodeView nodeId="tcode-unit-3d" payload={{ model: 'SR6', line: 'R09999', resetVersion: 1 }} />);
  expect(emulator.destroy).toHaveBeenCalledTimes(1);
  expect(emulator.create).toHaveBeenCalledTimes(2);
  expect(emulator.write).toHaveBeenLastCalledWith('R09999\n');

  view.rerender(<TCodeView nodeId="tcode-unit-3d" payload={{ model: 'SSR1', line: 'L02500', resetVersion: 1 }} />);
  expect(emulator.destroy).toHaveBeenCalledTimes(2);
  expect(emulator.create).toHaveBeenLastCalledWith(expect.any(HTMLElement), { model: 'SSR1' });
});

test('renders retained channel values without starting 3D in Bars mode', () => {
  setTCodeViewMode('tcode-unit-bars', 'bars');
  render(<TCodeView nodeId="tcode-unit-bars" compact payload={{ model: 'SR6', line: 'R09999I500', channels: { L0: 5000, R0: 9999 } }} />);
  expect(screen.getByText('L0')).toBeInTheDocument();
  expect(screen.getByText('5000')).toBeInTheDocument();
  expect(screen.getByText('R0')).toBeInTheDocument();
  expect(screen.getByText('9999')).toBeInTheDocument();
  expect(screen.queryByLabelText('SR6 3D TCode visualizer')).not.toBeInTheDocument();
  expect(emulator.create).not.toHaveBeenCalled();
});
