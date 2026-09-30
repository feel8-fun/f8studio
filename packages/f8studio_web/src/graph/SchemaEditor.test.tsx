import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import type { OperatorNode } from '../api/contracts';
import { SchemaEditor } from './SchemaEditor';

afterEach(cleanup);

const node: OperatorNode = {
  kind: 'operator', nodeId: 'script', name: 'Script', serviceId: 'engine', serviceClass: 'f8.pyengine',
  operatorClass: 'f8.python_script', enabled: true, portIds: {}, stateValues: {}, ports: [
    { dataSpec: null, stateSpec: null,  portId: 'state:input:code', name: 'code', runtimeName: 'code', kind: 'state', direction: 'input' },
    { dataSpec: null, stateSpec: null,  portId: 'state:output:code', name: 'code', runtimeName: 'code', kind: 'state', direction: 'output' },
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
  fireEvent.click(screen.getByText('Fields & ports'));
  fireEvent.click(screen.getByRole('button', { name: 'Add State fields' }));
  fireEvent.change(screen.getAllByRole('textbox', { name: 'State name' })[1]!, { target: { value: 'threshold' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply changes' }));
  expect(commit).toHaveBeenCalledWith([{ op: 'setOperatorSpec', nodeId: 'script', spec: expect.objectContaining({
    stateFields: expect.arrayContaining([expect.objectContaining({ name: 'threshold' })]),
  }), portRenames: {} }]);
});

test('submits stable port IDs when an interface is renamed', () => {
  const commit = vi.fn(async () => {});
  render(<SchemaEditor node={node} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Fields & ports'));
  fireEvent.change(screen.getByRole('textbox', { name: 'State name' }), { target: { value: 'scriptCode' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply changes' }));
  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    op: 'setOperatorSpec',
    portRenames: { 'state:input:code': 'scriptCode', 'state:output:code': 'scriptCode' },
  })]);
});

test('resets unsaved form and JSON edits together with pending port renames', () => {
  const commit = vi.fn(async () => {});
  render(<SchemaEditor node={node} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Fields & ports'));
  const reset = screen.getByRole('button', { name: 'Reset changes' });
  expect(reset).toBeDisabled();

  fireEvent.change(screen.getByRole('textbox', { name: 'State name' }), { target: { value: 'temporaryCode' } });
  expect(reset).toBeEnabled();
  fireEvent.click(reset);
  expect(screen.getByRole('textbox', { name: 'State name' })).toHaveValue('code');
  expect(screen.getByRole('textbox', { name: 'Node schema JSON' })).toHaveValue(JSON.stringify(node.spec, null, 2));
  expect(reset).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Settings for code' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'code label' }), { target: { value: 'Visible code' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply changes' }));
  expect(commit).toHaveBeenCalledWith([expect.objectContaining({ portRenames: {} })]);

  fireEvent.change(screen.getByRole('textbox', { name: 'Node schema JSON' }), { target: { value: '{invalid' } });
  fireEvent.click(screen.getByRole('button', { name: 'Reset changes' }));
  expect(screen.getByRole('textbox', { name: 'Node schema JSON' })).toHaveValue(JSON.stringify(node.spec, null, 2));
});

test('does not offer schema additions when the descriptor locks a collection', () => {
  render(<SchemaEditor node={{ ...node, spec: { ...node.spec, editPolicy: undefined } }} busy={false} commit={vi.fn(async () => {})} />);
  fireEvent.click(screen.getByText('Fields & ports'));
  expect(screen.queryByRole('button', { name: 'Add State fields' })).not.toBeInTheDocument();
});

test('edits data value type and command parameters through the form', () => {
  const commit = vi.fn(async () => {});
  const editable: OperatorNode = {
    ...node,
    spec: {
      ...node.spec,
      dataOutPorts: [{ name: 'result', valueSchema: { type: 'any' }, definitionProtected: false }],
      commands: [{ name: 'Run', params: [] }],
      editPolicy: {
        dataOutPorts: { canEditExisting: true },
        commands: { canEditExisting: true },
      },
    },
  };
  render(<SchemaEditor node={editable} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Fields & ports'));
  fireEvent.change(screen.getByRole('combobox', { name: 'result value type' }), { target: { value: 'number' } });
  fireEvent.click(screen.getByRole('button', { name: 'Settings for Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add parameter to Run' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Run parameter name' }), { target: { value: 'speed' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply changes' }));
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
  fireEvent.click(screen.getByText('Fields & ports'));
  fireEvent.click(screen.getByRole('button', { name: 'Settings for code' }));
  fireEvent.change(screen.getByRole('combobox', { name: 'code widget' }), { target: { value: 'toggle' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply changes' }));
  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    spec: expect.objectContaining({ stateFields: [expect.objectContaining({ control: { kind: 'toggle' } })] }),
  })]);
});

test('allows node presentation settings when the runtime interface is fixed', () => {
  const commit = vi.fn(async () => {});
  render(<SchemaEditor node={{ ...node, spec: { ...node.spec, editPolicy: undefined } }} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Fields & ports'));

  expect(screen.queryByRole('textbox', { name: 'State name' })).not.toBeInTheDocument();
  expect(screen.queryByRole('combobox', { name: 'code type' })).not.toBeInTheDocument();
  const visibility = screen.getByRole('button', { name: 'Show on node' });
  expect(visibility).toBeEnabled();
  fireEvent.click(visibility);
  fireEvent.click(screen.getByRole('button', { name: 'Settings for code' }));
  fireEvent.change(screen.getByRole('combobox', { name: 'code widget' }), { target: { value: 'textarea' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply changes' }));

  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    op: 'setOperatorSpec',
    spec: expect.objectContaining({ stateFields: [expect.objectContaining({
      name: 'code', showOnNode: true, control: { kind: 'textarea' },
    })] }),
  })]);
});

test('keeps data and exec details in per-item settings', () => {
  const compact: OperatorNode = {
    ...node,
    spec: {
      ...node.spec,
      dataOutPorts: [{ name: 'result', valueSchema: { type: 'string' } }],
      execInPorts: [{ name: 'run' }],
    },
  };
  render(<SchemaEditor node={compact} busy={false} commit={vi.fn(async () => {})} />);
  fireEvent.click(screen.getByText('Fields & ports'));
  expect(screen.queryByRole('textbox', { name: 'result description' })).not.toBeInTheDocument();
  expect(screen.queryByRole('textbox', { name: 'run label' })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Settings for result' }));
  expect(screen.getByRole('textbox', { name: 'result description' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Close settings' }));
  fireEvent.click(screen.getByRole('button', { name: 'Settings for run' }));
  expect(screen.getByRole('textbox', { name: 'run label' })).toBeVisible();
});

test('edits every common state property and applies directly from settings', () => {
  const commit = vi.fn(async () => {});
  render(<SchemaEditor node={node} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Fields & ports'));
  fireEvent.click(screen.getByRole('button', { name: 'Settings for code' }));
  const dialog = within(screen.getByRole('dialog', { name: 'code settings' }));

  fireEvent.change(dialog.getByRole('textbox', { name: 'State name' }), { target: { value: 'scriptCode' } });
  fireEvent.change(dialog.getByRole('combobox', { name: 'scriptCode type' }), { target: { value: 'boolean' } });
  fireEvent.change(dialog.getByRole('combobox', { name: 'scriptCode access' }), { target: { value: 'ro' } });
  fireEvent.click(dialog.getByRole('checkbox', { name: 'Show on node' }));
  fireEvent.click(dialog.getByRole('checkbox', { name: 'Value required' }));
  fireEvent.change(dialog.getByRole('textbox', { name: 'scriptCode label' }), { target: { value: 'Script code' } });
  fireEvent.change(dialog.getByRole('combobox', { name: 'scriptCode widget' }), { target: { value: 'toggle' } });
  fireEvent.click(dialog.getByRole('button', { name: 'Apply changes' }));

  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    portRenames: { 'state:input:code': 'scriptCode', 'state:output:code': 'scriptCode' },
    spec: expect.objectContaining({ stateFields: [expect.objectContaining({
      name: 'scriptCode', access: 'ro', valueSchema: { type: 'boolean' },
      showOnNode: true, valueRequired: true, label: 'Script code', control: { kind: 'toggle' },
    })] }),
  })]);
});

test('edits data, exec and command properties without returning to the item list', () => {
  const commit = vi.fn(async () => {});
  const editable: OperatorNode = {
    ...node,
    spec: {
      ...node.spec,
      dataOutPorts: [{ name: 'result', valueSchema: { type: 'string' } }],
      execInPorts: [{ name: 'run' }],
      commands: [{ name: 'Start', showOnNode: false }],
      editPolicy: {
        dataOutPorts: { canEditExisting: true },
        execInPorts: { canEditExisting: true },
        commands: { canEditExisting: true },
      },
    },
  };
  render(<SchemaEditor node={editable} busy={false} commit={commit} />);
  fireEvent.click(screen.getByText('Fields & ports'));

  fireEvent.click(screen.getByRole('button', { name: 'Settings for result' }));
  const data = within(screen.getByRole('dialog', { name: 'result settings' }));
  fireEvent.change(data.getByRole('textbox', { name: 'dataOutPorts name' }), { target: { value: 'output' } });
  fireEvent.change(data.getByRole('combobox', { name: 'output value type' }), { target: { value: 'number' } });
  fireEvent.click(data.getByRole('checkbox', { name: 'Show on node' }));
  fireEvent.click(data.getByRole('button', { name: 'Close settings' }));

  fireEvent.click(screen.getByRole('button', { name: 'Settings for run' }));
  const exec = within(screen.getByRole('dialog', { name: 'run settings' }));
  fireEvent.change(exec.getByRole('textbox', { name: 'execInPorts name' }), { target: { value: 'execute' } });
  fireEvent.click(exec.getByRole('button', { name: 'Close settings' }));

  fireEvent.click(screen.getByRole('button', { name: 'Settings for Start' }));
  const command = within(screen.getByRole('dialog', { name: 'Start settings' }));
  fireEvent.change(command.getByRole('textbox', { name: 'Command name' }), { target: { value: 'Launch' } });
  fireEvent.click(command.getByRole('checkbox', { name: 'Show on node' }));
  fireEvent.click(command.getByRole('button', { name: 'Apply changes' }));

  expect(commit).toHaveBeenCalledWith([expect.objectContaining({
    spec: expect.objectContaining({
      dataOutPorts: [expect.objectContaining({ name: 'output', valueSchema: { type: 'number' }, showOnNode: false })],
      execInPorts: [expect.objectContaining({ name: 'execute' })],
      commands: [expect.objectContaining({ name: 'Launch', showOnNode: true })],
    }),
  })]);
});
