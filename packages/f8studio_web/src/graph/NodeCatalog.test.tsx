import { fireEvent, render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';

import type { CatalogSnapshot } from '../api/contracts';
import { NodeCatalog } from './NodeCatalog';

const catalog: CatalogSnapshot = {
  services: [
    { specKind: 'service', serviceClass: 'test.engine', label: 'Engine' },
    { specKind: 'service', serviceClass: 'f8.pystudio', label: 'Studio' },
  ],
  operators: [
    { specKind: 'operator', serviceClass: 'test.engine', operatorClass: 'test.filter', label: 'Filter', paletteCategory: 'test.engine.signal' },
    { specKind: 'operator', serviceClass: 'test.engine', operatorClass: 'test.capture', label: 'Capture', paletteCategory: 'test.engine.input' },
    { specKind: 'operator', serviceClass: 'f8.pystudio', operatorClass: 'f8.viz.video', label: 'Video', paletteCategory: 'viz' },
  ],
};

test('groups operators by service or category and expands search matches', () => {
  const onAdd = vi.fn();
  render(<NodeCatalog catalog={catalog} projectServiceClasses={new Set(['test.engine'])} canAdd onAdd={onAdd} />);
  expect(screen.getByText('Engine', { selector: 'summary span' })).toBeInTheDocument();
  expect(screen.getByText('Studio', { selector: 'summary span' })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: 'Category' }));
  expect(screen.getByText('Signal', { selector: 'summary span' })).toBeInTheDocument();
  expect(screen.getByText('Inputs', { selector: 'summary span' })).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText('Search nodes'), { target: { value: 'Filter' } });
  fireEvent.click(screen.getByRole('button', { name: /Filter/ }));
  expect(onAdd).toHaveBeenCalledWith(catalog.operators[0]);
});
