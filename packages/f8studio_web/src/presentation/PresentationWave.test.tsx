import { cleanup, render } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import { PresentationWave } from './PresentationWave';

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

test('plots only points in the configured time window using the server color', () => {
  const context = {
    fillRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(),
    save: vi.fn(), rect: vi.fn(), clip: vi.fn(), restore: vi.fn(), fillText: vi.fn(),
    fillStyle: '', strokeStyle: '', lineWidth: 0,
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);

  render(<PresentationWave payload={{
    series: { x: [[500, 4], [1500, 2], [2000, 3]] },
    colors: { x: [244, 67, 54] }, windowMs: 1000, nowMs: 2000,
    minVal: 0, maxVal: 4, showLegend: true,
  }} />);

  expect(context.moveTo).toHaveBeenCalledWith(360, 120);
  expect(context.lineTo).toHaveBeenCalledWith(720, 60);
  expect(context.strokeStyle).toBe('rgb(244, 67, 54)');
  expect(context.fillText).toHaveBeenCalledWith('x', 8, 16);
});
