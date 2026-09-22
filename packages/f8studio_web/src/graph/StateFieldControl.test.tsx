import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import type { OperatorNode, StateSpec } from '../api/contracts';
import { StateFieldControl } from './StateFieldControl';

afterEach(cleanup);

const enabledField: StateSpec = {
  name: 'enabled', label: 'Enabled', access: 'rw', uiControl: 'toggle', showOnNode: true,
  valueSchema: { type: 'boolean', default: false },
};
const node: OperatorNode = {
  kind: 'operator', nodeId: 'operator', name: 'Operator', serviceId: 'service', serviceClass: 'f8.pyengine',
  operatorClass: 'test.operator',
  spec: {
    serviceClass: 'f8.pyengine', operatorClass: 'test.operator', label: 'Operator', specKind: 'operator',
    stateFields: [enabledField],
  },
  ports: [], stateValues: {}, enabled: true,
};

test('commits typed boolean values from a schema-driven control', () => {
  const commit = vi.fn();
  render(<StateFieldControl node={node} field={enabledField} disabled={false} onCommit={commit} />);

  fireEvent.click(screen.getByRole('checkbox', { name: 'Enabled' }));

  expect(commit).toHaveBeenCalledWith(true);
});

test('renders upstream-driven state as read-only', () => {
  render(<StateFieldControl node={node} field={enabledField} disabled={false} connected onCommit={vi.fn()} />);

  expect(screen.getByRole('checkbox', { name: 'Enabled' })).toBeDisabled();
  expect(screen.getByText('Upstream')).toBeInTheDocument();
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

test('keeps the compact numeric input mounted when upstream drives its value', () => {
  const numberField: StateSpec = {
    name: 'level', label: 'Level', access: 'rw', valueSchema: { type: 'number', default: 0.5 },
  };
  const commit = vi.fn();
  render(<StateFieldControl
    node={{ ...node, spec: { ...node.spec, stateFields: [numberField] } }}
    field={numberField}
    compact
    connected
    disabled={false}
    onCommit={commit}
  />);

  const input = screen.getByRole('spinbutton');
  expect(input).toHaveAttribute('readonly');
  fireEvent.change(input, { target: { value: '0.75' } });
  fireEvent.blur(input);
  expect(commit).not.toHaveBeenCalled();
});

test('does not commit an unchanged default value when an editor loses focus', () => {
  const numberField: StateSpec = {
    name: 'interval', label: 'Interval', access: 'rw', valueSchema: { type: 'number', default: 8 },
  };
  const commit = vi.fn();
  render(<StateFieldControl
    node={{ ...node, spec: { ...node.spec, stateFields: [numberField] } }}
    field={numberField}
    disabled={false}
    onCommit={commit}
  />);

  fireEvent.blur(screen.getByRole('spinbutton', { name: 'Interval' }));

  expect(commit).not.toHaveBeenCalled();
});

test('commits typed arrays from a dynamic multiselect control', () => {
  const poolField: StateSpec = {
    name: 'available', access: 'ro', valueSchema: { type: 'array', default: ['left', 'right'] },
  };
  const selectedField: StateSpec = {
    name: 'selected', label: 'Selected', access: 'rw', uiControl: 'multiselect[available]',
    valueSchema: { type: 'array', default: ['left'] },
  };
  const commit = vi.fn();
  render(<StateFieldControl
    node={{ ...node, spec: { ...node.spec, stateFields: [poolField, selectedField] } }}
    field={selectedField}
    disabled={false}
    onCommit={commit}
  />);
  const listbox = screen.getByRole('listbox', { name: 'Selected' });
  for (const option of listbox.querySelectorAll('option')) option.selected = true;

  fireEvent.change(listbox);

  expect(commit).toHaveBeenCalledWith(['left', 'right']);
});

test('uses a fixed single-line editor for compact wrapline state', () => {
  const expressionField: StateSpec = {
    name: 'code', label: 'Expr', access: 'rw', uiControl: 'wrapline[python]',
    valueSchema: { type: 'string', default: 'value * 2' },
  };
  render(<StateFieldControl
    node={{ ...node, spec: { ...node.spec, stateFields: [expressionField] } }}
    field={expressionField}
    compact
    disabled={false}
    onCommit={vi.fn()}
  />);

  expect(screen.getByRole('textbox').tagName).toBe('INPUT');
  expect(document.querySelector('textarea')).toBeNull();
});
