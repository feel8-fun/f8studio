import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import type { OperatorNode } from '../api/contracts';
import { SchemaEditor } from './SchemaEditor';

afterEach(cleanup);

const node: OperatorNode = {
  kind: 'operator', nodeId: 'script', name: 'Script', serviceId: 'engine', serviceClass: 'f8.pyengine',
  operatorClass: 'f8.python_script', enabled: true, stateValues: {}, ports: [
    { portId: 'state:input:code', name: 'code', runtimeName: 'code', kind: 'state', direction: 'input' },
    { portId: 'state:output:code', name: 'code', runtimeName: 'code', kind: 'state', direction: 'output' },
  ],
  spec: {
    specKind: 'operator', serviceClass: 'f8.pyengine', operatorClass: 'f8.python_script', label: 'Script',
    stateFields: [{ name: 'code', access: 'rw', valueSchema: { type: 'string' }, showOnNode: false }],
    editPolicy: { stateFields: { canAdd: true, canDelete: true, canEditExisting: true } },
  },
};

test('edits dynamic state fields without sending derived ports', async () => {
  const commit = vi.fn(async () => {});
  render(<SchemaEditor node={node} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Schema'));
  fireEvent.click(screen.getByRole('button', { name: 'Add State fields' }));
  fireEvent.change(screen.getAllByRole('textbox', { name: 'State name' })[1]!, { target: { value: 'threshold' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply schema' }));
  expect(commit).toHaveBeenCalledWith([{ op: 'setOperatorSpec', nodeId: 'script', spec: expect.objectContaining({
    stateFields: expect.arrayContaining([expect.objectContaining({ name: 'threshold' })]),
  }), portRenames: {} }]);
});

test('submits stable port IDs when an interface is renamed', () => {
  const commit = vi.fn(async () => {});
  render(<SchemaEditor node={node} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Schema'));
  fireEvent.change(screen.getByRole('textbox', { name: 'State name' }), { target: { value: 'scriptCode' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply schema' }));
  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    op: 'setOperatorSpec',
    portRenames: { 'state:input:code': 'scriptCode', 'state:output:code': 'scriptCode' },
  })]);
});

test('does not offer schema additions when the descriptor locks a collection', () => {
  render(<SchemaEditor node={{ ...node, spec: { ...node.spec, editPolicy: undefined } }} busy={false} commit={vi.fn(async () => {})} />);
  fireEvent.click(screen.getByText('Schema'));
  expect(screen.queryByRole('button', { name: 'Add State fields' })).not.toBeInTheDocument();
});

test('edits data value type and command parameters through the form', () => {
  const commit = vi.fn(async () => {});
  const editable: OperatorNode = {
    ...node,
    spec: {
      ...node.spec,
      dataOutPorts: [{ name: 'result', valueSchema: { type: 'any' }, required: false }],
      commands: [{ name: 'Run', params: [] }],
      editPolicy: {
        dataOutPorts: { canEditExisting: true },
        commands: { canEditExisting: true },
      },
    },
  };
  render(<SchemaEditor node={editable} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Schema'));
  fireEvent.change(screen.getByRole('combobox', { name: 'result value type' }), { target: { value: 'number' } });
  fireEvent.click(screen.getByRole('button', { name: 'Add parameter to Run' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Run parameter name' }), { target: { value: 'speed' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply schema' }));
  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    spec: expect.objectContaining({
      dataOutPorts: [expect.objectContaining({ valueSchema: { type: 'number' } })],
      commands: [expect.objectContaining({ params: [expect.objectContaining({ name: 'speed' })] })],
    }),
  })]);
});

test('changing control kind clears incompatible control options', () => {
  const commit = vi.fn(async () => {});
  const editable: OperatorNode = {
    ...node,
    spec: {
      ...node.spec,
      stateFields: [{ ...node.spec.stateFields![0]!, control: { kind: 'select', optionsFromState: 'choices' } }],
    },
  };
  render(<SchemaEditor node={editable} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Schema'));
  fireEvent.change(screen.getByRole('combobox', { name: 'code control' }), { target: { value: 'toggle' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply schema' }));
  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    spec: expect.objectContaining({ stateFields: [expect.objectContaining({ control: { kind: 'toggle' } })] }),
  })]);
});
