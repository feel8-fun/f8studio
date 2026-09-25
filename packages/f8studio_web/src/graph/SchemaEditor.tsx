import { Braces, Plus, Trash2 } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';

import type {
  CollectionEditPolicy,
  CommandParamSpec,
  CommandSpec,
  DataPortSpec,
  GraphNode,
  GraphOperation,
  PortDirection,
  PortKind,
  OperatorSpec,
  ServiceSpec,
  StateSpec,
  UiControlSpec,
  ValueSchema,
} from '../api/contracts';

type Collection = 'stateFields' | 'commands' | 'dataInPorts' | 'dataOutPorts' | 'execInPorts' | 'execOutPorts';
type Spec = ServiceSpec | OperatorSpec;

function nextName(names: readonly string[], prefix: string): string {
  let index = 1;
  while (names.includes(`${prefix}${index}`)) index += 1;
  return `${prefix}${index}`;
}

function valueType(schema: ValueSchema): string {
  return schema.type ?? 'any';
}

function SchemaSection({ title, policy, children, onAdd, busy }: {
  readonly title: string;
  readonly policy: CollectionEditPolicy | undefined;
  readonly children: ReactNode;
  readonly onAdd: () => void;
  readonly busy: boolean;
}) {
  return <section className="schema-section">
    <div className="schema-section-heading"><strong>{title}</strong>{policy?.canAdd === true &&
      <button type="button" title={`Add ${title}`} aria-label={`Add ${title}`} disabled={busy} onClick={onAdd}><Plus size={13} /></button>}</div>
    {children}
  </section>;
}

export function SchemaEditor({ node, busy, commit }: {
  readonly node: GraphNode;
  readonly busy: boolean;
  readonly commit: (operations: readonly GraphOperation[]) => Promise<void>;
}) {
  const [draft, setDraft] = useState<Spec>(node.spec);
  const [text, setText] = useState(JSON.stringify(node.spec, null, 2));
  const [portRenames, setPortRenames] = useState<Readonly<Record<string, string>>>({});
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setDraft(node.spec);
    setText(JSON.stringify(node.spec, null, 2));
    setPortRenames({});
    setError(null);
  }, [node]);

  const update = (next: Spec): void => {
    setDraft(next);
    setText(JSON.stringify(next, null, 2));
    setError(null);
  };
  const trackRename = (kind: PortKind, direction: PortDirection | null, current: string, next: string): void => {
    setPortRenames((previous) => {
      const updated = { ...previous };
      for (const port of node.ports) {
        if (port.kind === kind && (direction === null || port.direction === direction) &&
          (port.name === current || previous[port.portId] === current)) {
          updated[port.portId] = next;
        }
      }
      return updated;
    });
  };
  const policy = (collection: Collection): CollectionEditPolicy | undefined => draft.editPolicy?.[collection];
  const canEdit = (collection: Collection): boolean => !busy && policy(collection)?.canEditExisting === true;
  const canDelete = (collection: Collection, protectedItem = false): boolean => !busy && !protectedItem && policy(collection)?.canDelete === true;
  const setStates = (fields: readonly StateSpec[]): void => update({ ...draft, stateFields: fields });
  const setCommands = (commands: readonly CommandSpec[]): void => update({ ...draft, commands });
  const setData = (key: 'dataInPorts' | 'dataOutPorts', ports: readonly DataPortSpec[]): void => update({ ...draft, [key]: ports });
  const setExec = (key: 'execInPorts' | 'execOutPorts', names: readonly string[]): void => {
    if (draft.specKind === 'operator') update({ ...draft, [key]: names });
  };

  const save = async (): Promise<void> => {
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error('Schema must be an object');
      await commit([node.kind === 'operator'
        ? { op: 'setOperatorSpec', nodeId: node.nodeId, spec: parsed as OperatorSpec, portRenames }
        : { op: 'setServiceSpec', nodeId: node.nodeId, spec: parsed as ServiceSpec, portRenames }]);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : 'Schema update failed');
    }
  };

  return <details className="node-schema-editor">
    <summary><Braces size={14} />Schema</summary>
    <SchemaSection title="State fields" policy={policy('stateFields')} busy={busy} onAdd={() => {
      const fields = draft.stateFields ?? [];
      setStates([...fields, { name: nextName(fields.map((field) => field.name), 'state'), access: 'rw', valueSchema: { type: 'string' }, showOnNode: false }]);
    }}>
      {(draft.stateFields ?? []).map((field, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main">
          <input aria-label="State name" title="Runtime field name" value={field.name} disabled={!canEdit('stateFields') || field.editPolicy?.canRename === false}
            onChange={(event) => {
              trackRename('state', null, field.name, event.target.value);
              setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
            }} />
          <select aria-label={`${field.name} type`} value={valueType(field.valueSchema)} disabled={!canEdit('stateFields') || field.editPolicy?.canEditValueSchema === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, valueSchema: { type: event.target.value } } : item))}>
            {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
          </select>
          <button type="button" title={`Delete ${field.name}`} aria-label={`Delete ${field.name}`} disabled={!canDelete('stateFields', field.editPolicy?.canRename === false)}
            onClick={() => setStates((draft.stateFields ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>
        </div>
        <div className="schema-item-options">
          <select aria-label={`${field.name} access`} value={field.access} disabled={!canEdit('stateFields') || field.editPolicy?.canEditAccess === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, access: event.target.value as StateSpec['access'] } : item))}>
            <option value="rw">Read/write</option><option value="ro">Read only</option><option value="wo">Write only</option>
          </select>
          <label><input type="checkbox" checked={field.showOnNode === true} disabled={!canEdit('stateFields')}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, showOnNode: event.target.checked } : item))} />Node</label>
          <label><input type="checkbox" checked={field.required === true} disabled={!canEdit('stateFields') || field.editPolicy?.canEditRequired === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, required: event.target.checked } : item))} />Required</label>
        </div>
        <input className="schema-detail-input" aria-label={`${field.name} label`} placeholder="Display label" value={field.label ?? ''} disabled={!canEdit('stateFields')}
          onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, label: event.target.value } : item))} />
        <select className="schema-detail-input" aria-label={`${field.name} control`} value={field.control?.kind ?? 'auto'} disabled={!canEdit('stateFields')}
          onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
            ...item, control: { kind: event.target.value as UiControlSpec['kind'] },
          } : item))}>
          {['auto', 'text', 'textarea', 'code', 'toggle', 'slider', 'select', 'multiselect', 'dial', 'button', 'custom'].map((kind) =>
            <option key={kind} value={kind}>{kind}</option>)}
        </select>
        {(field.control?.kind === 'select' || field.control?.kind === 'multiselect') &&
          <input className="schema-detail-input" aria-label={`${field.name} options source`} placeholder="Options state field" value={field.control.optionsFromState ?? ''}
            disabled={!canEdit('stateFields')} onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
              ...item, control: { ...item.control!, optionsFromState: event.target.value },
            } : item))} />}
        {(field.control?.kind === 'code' || field.control?.kind === 'textarea') &&
          <input className="schema-detail-input" aria-label={`${field.name} language`} placeholder="Language" value={field.control.language ?? ''}
            disabled={!canEdit('stateFields')} onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
              ...item, control: { ...item.control!, language: event.target.value },
            } : item))} />}
        {field.control?.kind === 'custom' &&
          <input className="schema-detail-input" aria-label={`${field.name} renderer key`} placeholder="Renderer key" value={field.control.rendererKey ?? ''}
            disabled={!canEdit('stateFields')} onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
              ...item, control: { ...item.control!, rendererKey: event.target.value },
            } : item))} />}
      </div>)}
    </SchemaSection>
    {(['dataInPorts', 'dataOutPorts'] as const).map((key) => <SchemaSection key={key} title={key === 'dataInPorts' ? 'Data inputs' : 'Data outputs'} policy={policy(key)} busy={busy} onAdd={() => {
      const ports = draft[key] ?? [];
      setData(key, [...ports, { name: nextName(ports.map((port) => port.name), 'data'), valueSchema: { type: 'any' }, required: false, showOnNode: true }]);
    }}>
      {(draft[key] ?? []).map((port, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main">
          <input aria-label={`${key} name`} value={port.name} disabled={!canEdit(key)}
            onChange={(event) => {
              trackRename('data', key === 'dataInPorts' ? 'input' : 'output', port.name, event.target.value);
              setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
            }} />
          {(port.payload?.kind ?? port.payloadKind ?? 'json') === 'json'
            ? <select aria-label={`${port.name} value type`} value={valueType(port.valueSchema)} disabled={!canEdit(key)}
              onChange={(event) => setData(key, (draft[key] ?? []).map((item, i) => {
                if (i !== index) return item;
                const valueSchema: ValueSchema = { type: event.target.value };
                return { ...item, valueSchema, ...(item.payload === undefined ? {} : { payload: { ...item.payload, valueSchema } }) };
              }))}>
              {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
            </select>
            : <span className="schema-kind">{port.payload?.kind ?? port.payloadKind}</span>}
          <button type="button" title={`Delete ${port.name}`} aria-label={`Delete ${port.name}`} disabled={!canDelete(key, port.required !== false)}
            onClick={() => setData(key, (draft[key] ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>
        </div>
        <label className="schema-item-options"><input type="checkbox" checked={port.showOnNode !== false} disabled={!canEdit(key)}
          onChange={(event) => setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, showOnNode: event.target.checked } : item))} />Show on node</label>
      </div>)}
    </SchemaSection>)}
    {draft.specKind === 'operator' && (['execInPorts', 'execOutPorts'] as const).map((key) => <SchemaSection key={key}
      title={key === 'execInPorts' ? 'Exec inputs' : 'Exec outputs'} policy={policy(key)} busy={busy} onAdd={() => {
        const names = draft[key] ?? [];
        setExec(key, [...names, nextName(names, 'exec')]);
      }}>
      {(draft[key] ?? []).map((name, index) => <div className="schema-item-main" key={index}>
        <input aria-label={`${key} name`} value={name} disabled={!canEdit(key)}
          onChange={(event) => {
            trackRename('exec', key === 'execInPorts' ? 'input' : 'output', name, event.target.value);
            setExec(key, (draft[key] ?? []).map((item, i) => i === index ? event.target.value : item));
          }} />
        <button type="button" title={`Delete ${name}`} aria-label={`Delete ${name}`} disabled={!canDelete(key)}
          onClick={() => setExec(key, (draft[key] ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>
      </div>)}
    </SchemaSection>)}
    <SchemaSection title="Commands" policy={policy('commands')} busy={busy} onAdd={() => {
      const commands = draft.commands ?? [];
      setCommands([...commands, { name: nextName(commands.map((command) => command.name), 'command'), required: false, showOnNode: false }]);
    }}>
      {(draft.commands ?? []).map((command, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main"><input aria-label="Command name" value={command.name} disabled={!canEdit('commands')}
          onChange={(event) => {
            trackRename('command', null, command.name, event.target.value);
            setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
          }} />
          <button type="button" title={`Delete ${command.name}`} aria-label={`Delete ${command.name}`} disabled={!canDelete('commands', command.required === true)}
            onClick={() => setCommands((draft.commands ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button></div>
        <label className="schema-item-options"><input type="checkbox" checked={command.showOnNode === true} disabled={!canEdit('commands')}
          onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, showOnNode: event.target.checked } : item))} />Show on node</label>
        <div className="schema-params-heading"><span>Parameters</span><button type="button" title={`Add parameter to ${command.name}`} aria-label={`Add parameter to ${command.name}`}
          disabled={!canEdit('commands')} onClick={() => {
            const params = command.params ?? [];
            setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: [...params, { name: nextName(params.map((param) => param.name), 'arg'), valueSchema: { type: 'string' }, required: false }],
            } : item));
          }}><Plus size={12} /></button></div>
        {(command.params ?? []).map((param, paramIndex) => <div className="schema-item-main schema-param" key={paramIndex}>
          <input aria-label={`${command.name} parameter name`} value={param.name} disabled={!canEdit('commands')}
            onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: (item.params ?? []).map((value, j): CommandParamSpec => j === paramIndex ? { ...value, name: event.target.value } : value),
            } : item))} />
          <select aria-label={`${param.name} parameter type`} value={valueType(param.valueSchema)} disabled={!canEdit('commands')}
            onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: (item.params ?? []).map((value, j) => j === paramIndex ? { ...value, valueSchema: { type: event.target.value } } : value),
            } : item))}>
            {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
          </select>
          <button type="button" title={`Delete ${param.name}`} aria-label={`Delete ${param.name}`} disabled={!canEdit('commands')}
            onClick={() => setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: (item.params ?? []).filter((_, j) => j !== paramIndex),
            } : item))}><Trash2 size={12} /></button>
        </div>)}
      </div>)}
    </SchemaSection>
    <details className="schema-advanced"><summary>Advanced JSON</summary><textarea aria-label="Node schema JSON" value={text} disabled={busy} spellCheck={false}
      onChange={(event) => {
        setText(event.target.value);
        setError(null);
        try {
          const parsed: unknown = JSON.parse(event.target.value);
          if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) && 'specKind' in parsed) setDraft(parsed as Spec);
        } catch { /* Partial JSON is allowed while typing. */ }
      }} /></details>
    {error !== null && <p role="alert">{error}</p>}
    <button className="command-button" type="button" disabled={busy || text === JSON.stringify(node.spec, null, 2)} onClick={() => void save()}>Apply schema</button>
  </details>;
}
