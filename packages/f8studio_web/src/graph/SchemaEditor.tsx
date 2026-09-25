import { Braces, Eye, EyeOff, Plus, Settings2, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

import type {
  CollectionEditPolicy,
  CommandParamSpec,
  CommandSpec,
  DataPortSpec,
  ExecPortSpec,
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

function SettingsDialog({ title, children, onClose }: {
  readonly title: string;
  readonly children: ReactNode;
  readonly onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialogRef.current?.querySelector('button')?.focus();
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      } else if (event.key === 'Tab' && dialogRef.current !== null) {
        const controls = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled)')];
        const first = controls[0];
        const last = controls.at(-1);
        if (first === undefined || last === undefined) return;
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      previousFocus?.focus();
    };
  }, [onClose]);
  return createPortal(<div className="schema-dialog-backdrop" onMouseDown={(event) => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <section className="schema-dialog" ref={dialogRef} role="dialog" aria-modal="true" aria-label={title}>
      <header><strong>{title}</strong><button type="button" aria-label="Close settings" title="Close settings" onClick={onClose}><X size={15} /></button></header>
      <div className="schema-dialog-content">{children}</div>
    </section>
  </div>, document.body);
}

function VisibilityButton({ visible, onChange, busy }: {
  readonly visible: boolean;
  readonly onChange: (visible: boolean) => void;
  readonly busy: boolean;
}) {
  return <button type="button" className={visible ? 'schema-visibility active' : 'schema-visibility'}
    title={visible ? 'Visible on node' : 'Hidden on node'} aria-label={visible ? 'Hide from node' : 'Show on node'}
    aria-pressed={visible} disabled={busy} onClick={() => onChange(!visible)}>
    {visible ? <Eye size={14} /> : <EyeOff size={14} />}
  </button>;
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
  const [activeSettings, setActiveSettings] = useState<string | null>(null);
  const closeSettings = useCallback(() => setActiveSettings(null), []);
  useEffect(() => {
    setDraft(node.spec);
    setText(JSON.stringify(node.spec, null, 2));
    setPortRenames({});
    setError(null);
    setActiveSettings(null);
  }, [node]);

  const update = (next: Spec): void => {
    setDraft(next);
    setText(JSON.stringify(next, null, 2));
    setError(null);
  };
  const reset = (): void => {
    setDraft(node.spec);
    setText(JSON.stringify(node.spec, null, 2));
    setPortRenames({});
    setError(null);
    setActiveSettings(null);
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
  const allowsEdit = (collection: Collection): boolean => policy(collection)?.canEditExisting === true;
  const canEdit = (collection: Collection): boolean => !busy && allowsEdit(collection);
  const canDelete = (collection: Collection, protectedItem = false): boolean => !busy && !protectedItem && policy(collection)?.canDelete === true;
  const setStates = (fields: readonly StateSpec[]): void => update({ ...draft, stateFields: fields });
  const setCommands = (commands: readonly CommandSpec[]): void => update({ ...draft, commands });
  const setData = (key: 'dataInPorts' | 'dataOutPorts', ports: readonly DataPortSpec[]): void => update({ ...draft, [key]: ports });
  const setExec = (key: 'execInPorts' | 'execOutPorts', ports: readonly ExecPortSpec[]): void => {
    if (draft.specKind === 'operator') update({ ...draft, [key]: ports });
  };

  const save = async (): Promise<void> => {
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error('Node interface must be an object');
      await commit([node.kind === 'operator'
        ? { op: 'setOperatorSpec', nodeId: node.nodeId, spec: parsed as OperatorSpec, portRenames }
        : { op: 'setServiceSpec', nodeId: node.nodeId, spec: parsed as ServiceSpec, portRenames }]);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : 'Interface update failed');
    }
  };
  const renderActions = (): ReactNode => <>
    {error !== null && <p role="alert">{error}</p>}
    <div className="schema-actions">
      <button className="command-button" type="button" disabled={busy || (text === JSON.stringify(node.spec, null, 2) && error === null)} onClick={reset}>Reset changes</button>
      <button className="command-button" type="button" disabled={busy || text === JSON.stringify(node.spec, null, 2)} onClick={() => void save()}>Apply changes</button>
    </div>
  </>;

  return <details className="node-schema-editor">
    <summary><Braces size={14} />Fields &amp; ports</summary>
    <SchemaSection title="State fields" policy={policy('stateFields')} busy={busy} onAdd={() => {
      const fields = draft.stateFields ?? [];
      setStates([...fields, { name: nextName(fields.map((field) => field.name), 'state'), access: 'rw', valueSchema: { type: 'string' }, showOnNode: false }]);
    }}>
      {(draft.stateFields ?? []).map((field, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main">
          {allowsEdit('stateFields') ? <input aria-label="State name" title="Runtime field name" value={field.name} disabled={!canEdit('stateFields') || field.editPolicy?.canRename === false}
            onChange={(event) => {
              trackRename('state', null, field.name, event.target.value);
              setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
            }} /> : <span className="schema-item-name">{field.name}</span>}
          {allowsEdit('stateFields') ? <select className="schema-type-select" aria-label={`${field.name} type`} value={valueType(field.valueSchema)} disabled={!canEdit('stateFields') || field.editPolicy?.canEditValueSchema === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, valueSchema: { type: event.target.value } } : item))}>
            {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
          </select> : <span className="schema-kind">{valueType(field.valueSchema)}</span>}
          {allowsEdit('stateFields') ? <select className="schema-access-select" aria-label={`${field.name} access`} title="State access" value={field.access} disabled={!canEdit('stateFields') || field.editPolicy?.canEditAccess === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, access: event.target.value as StateSpec['access'] } : item))}>
            <option value="rw">RW</option><option value="ro">RO</option><option value="wo">WO</option>
          </select> : <span className="schema-access-label" title="State access">{field.access.toUpperCase()}</span>}
          <VisibilityButton visible={field.showOnNode === true} busy={busy} onChange={(visible) =>
            setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, showOnNode: visible } : item))} />
          <button type="button" title={`Settings for ${field.name}`} aria-label={`Settings for ${field.name}`} onClick={() => setActiveSettings(`state:${index}`)}><Settings2 size={14} /></button>
          {allowsEdit('stateFields') && <button type="button" title={`Delete ${field.name}`} aria-label={`Delete ${field.name}`} disabled={!canDelete('stateFields', field.editPolicy?.canRename === false)}
            onClick={() => setStates((draft.stateFields ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>}
        </div>
        {activeSettings === `state:${index}` && <SettingsDialog title={`${field.name} settings`} onClose={closeSettings}>
          <label className="schema-detail-field">Name{allowsEdit('stateFields') ? <input aria-label="State name" value={field.name} disabled={!canEdit('stateFields') || field.editPolicy?.canRename === false}
            onChange={(event) => {
              trackRename('state', null, field.name, event.target.value);
              setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
            }} /> : <output>{field.name}</output>}</label>
          <label className="schema-detail-field">Type{allowsEdit('stateFields') ? <select aria-label={`${field.name} type`} value={valueType(field.valueSchema)} disabled={!canEdit('stateFields') || field.editPolicy?.canEditValueSchema === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, valueSchema: { type: event.target.value } } : item))}>
            {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
          </select> : <output>{valueType(field.valueSchema)}</output>}</label>
          <label className="schema-detail-field">Access{allowsEdit('stateFields') ? <select aria-label={`${field.name} access`} value={field.access} disabled={!canEdit('stateFields') || field.editPolicy?.canEditAccess === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, access: event.target.value as StateSpec['access'] } : item))}>
            <option value="rw">Read/write</option><option value="ro">Read only</option><option value="wo">Write only</option>
          </select> : <output>{field.access.toUpperCase()}</output>}</label>
          <label className="schema-dialog-check"><input type="checkbox" checked={field.showOnNode === true} disabled={busy}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, showOnNode: event.target.checked } : item))} />Show on node</label>
          <label className="schema-detail-field">Label<input aria-label={`${field.name} label`} value={field.label ?? ''} disabled={busy}
          onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, label: event.target.value } : item))} /></label>
          {allowsEdit('stateFields') ? <label className="schema-dialog-check"><input type="checkbox" checked={field.valueRequired === true} disabled={!canEdit('stateFields') || field.editPolicy?.canEditValueRequired === false}
            onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? { ...item, valueRequired: event.target.checked } : item))} />Value required</label>
            : field.valueRequired === true && <span className="schema-item-meta">Required value</span>}
          <label className="schema-detail-field">Widget<select aria-label={`${field.name} widget`} value={field.control?.kind ?? 'auto'} disabled={busy}
          onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
            ...item, control: { kind: event.target.value as UiControlSpec['kind'] },
          } : item))}>
          {['auto', 'text', 'textarea', 'code', 'toggle', 'slider', 'select', 'multiselect', 'dial', 'button', 'custom'].map((kind) =>
            <option key={kind} value={kind}>{kind}</option>)}
          </select></label>
          {(field.control?.kind === 'select' || field.control?.kind === 'multiselect') &&
          <label className="schema-detail-field">Options state field<input aria-label={`${field.name} options source`} value={field.control.optionsFromState ?? ''}
            disabled={busy} onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
              ...item, control: { ...item.control!, optionsFromState: event.target.value },
            } : item))} /></label>}
          {(field.control?.kind === 'code' || field.control?.kind === 'textarea') &&
          <label className="schema-detail-field">Language<input aria-label={`${field.name} language`} value={field.control.language ?? ''}
            disabled={busy} onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
              ...item, control: { ...item.control!, language: event.target.value },
            } : item))} /></label>}
          {field.control?.kind === 'custom' &&
          <label className="schema-detail-field">Renderer key<input aria-label={`${field.name} renderer key`} value={field.control.rendererKey ?? ''}
            disabled={busy} onChange={(event) => setStates((draft.stateFields ?? []).map((item, i) => i === index ? {
              ...item, control: { ...item.control!, rendererKey: event.target.value },
            } : item))} /></label>}
          {renderActions()}
        </SettingsDialog>}
      </div>)}
    </SchemaSection>
    {(['dataInPorts', 'dataOutPorts'] as const).map((key) => <SchemaSection key={key} title={key === 'dataInPorts' ? 'Data inputs' : 'Data outputs'} policy={policy(key)} busy={busy} onAdd={() => {
      const ports = draft[key] ?? [];
      setData(key, [...ports, { name: nextName(ports.map((port) => port.name), 'data'), valueSchema: { type: 'any' }, definitionProtected: false, showOnNode: true }]);
    }}>
      {(draft[key] ?? []).map((port, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main">
          {allowsEdit(key) ? <input aria-label={`${key} name`} value={port.name} disabled={!canEdit(key)}
            onChange={(event) => {
              trackRename('data', key === 'dataInPorts' ? 'input' : 'output', port.name, event.target.value);
              setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
            }} /> : <span className="schema-item-name">{port.name}</span>}
          {!allowsEdit(key) ? <span className="schema-kind">{(port.payload?.kind ?? port.payloadKind ?? 'json') === 'json' ? valueType(port.valueSchema) : port.payload?.kind ?? port.payloadKind}</span>
            : (port.payload?.kind ?? port.payloadKind ?? 'json') === 'json'
            ? <select className="schema-type-select" aria-label={`${port.name} value type`} value={valueType(port.valueSchema)} disabled={!canEdit(key)}
              onChange={(event) => setData(key, (draft[key] ?? []).map((item, i) => {
                if (i !== index) return item;
                const valueSchema: ValueSchema = { type: event.target.value };
                return { ...item, valueSchema, ...(item.payload === undefined ? {} : { payload: { ...item.payload, valueSchema } }) };
              }))}>
              {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
            </select>
            : <span className="schema-kind">{port.payload?.kind ?? port.payloadKind}</span>}
          <VisibilityButton visible={port.showOnNode !== false} busy={busy} onChange={(visible) =>
            setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, showOnNode: visible } : item))} />
          <button type="button" title={`Settings for ${port.name}`} aria-label={`Settings for ${port.name}`} onClick={() => setActiveSettings(`${key}:${index}`)}><Settings2 size={14} /></button>
          {allowsEdit(key) && <button type="button" title={`Delete ${port.name}`} aria-label={`Delete ${port.name}`} disabled={!canDelete(key, port.definitionProtected !== false)}
            onClick={() => setData(key, (draft[key] ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>}
        </div>
        {activeSettings === `${key}:${index}` && <SettingsDialog title={`${port.name} settings`} onClose={closeSettings}>
          <label className="schema-detail-field">Name{allowsEdit(key) ? <input aria-label={`${key} name`} value={port.name} disabled={!canEdit(key)}
            onChange={(event) => {
              trackRename('data', key === 'dataInPorts' ? 'input' : 'output', port.name, event.target.value);
              setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
            }} /> : <output>{port.name}</output>}</label>
          <label className="schema-detail-field">Type{allowsEdit(key) && (port.payload?.kind ?? port.payloadKind ?? 'json') === 'json' ? <select aria-label={`${port.name} value type`} value={valueType(port.valueSchema)} disabled={!canEdit(key)}
            onChange={(event) => setData(key, (draft[key] ?? []).map((item, i) => {
              if (i !== index) return item;
              const valueSchema: ValueSchema = { type: event.target.value };
              return { ...item, valueSchema, ...(item.payload === undefined ? {} : { payload: { ...item.payload, valueSchema } }) };
            }))}>
            {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
          </select> : <output>{(port.payload?.kind ?? port.payloadKind ?? 'json') === 'json' ? valueType(port.valueSchema) : port.payload?.kind ?? port.payloadKind}</output>}</label>
          <label className="schema-dialog-check"><input type="checkbox" checked={port.showOnNode !== false} disabled={busy}
            onChange={(event) => setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, showOnNode: event.target.checked } : item))} />Show on node</label>
          <label className="schema-detail-field">Description<input aria-label={`${port.name} description`} value={typeof port.description === 'string' ? port.description : ''} disabled={busy}
            onChange={(event) => setData(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, description: event.target.value } : item))} /></label>
          {renderActions()}
        </SettingsDialog>}
      </div>)}
    </SchemaSection>)}
    {draft.specKind === 'operator' && (['execInPorts', 'execOutPorts'] as const).map((key) => <SchemaSection key={key}
      title={key === 'execInPorts' ? 'Exec inputs' : 'Exec outputs'} policy={policy(key)} busy={busy} onAdd={() => {
        const ports = draft[key] ?? [];
        setExec(key, [...ports, { name: nextName(ports.map((port) => port.name), 'exec'), definitionProtected: false }]);
      }}>
      {(draft[key] ?? []).map((port, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main">{allowsEdit(key) ? <input aria-label={`${key} name`} value={port.name} disabled={!canEdit(key)}
          onChange={(event) => {
            trackRename('exec', key === 'execInPorts' ? 'input' : 'output', port.name, event.target.value);
            setExec(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
          }} /> : <span className="schema-item-name">{port.name}</span>}
        <button type="button" title={`Settings for ${port.name}`} aria-label={`Settings for ${port.name}`} onClick={() => setActiveSettings(`${key}:${index}`)}><Settings2 size={14} /></button>
        {allowsEdit(key) && <button type="button" title={`Delete ${port.name}`} aria-label={`Delete ${port.name}`} disabled={!canDelete(key, port.definitionProtected === true)}
          onClick={() => setExec(key, (draft[key] ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>}</div>
        {activeSettings === `${key}:${index}` && <SettingsDialog title={`${port.name} settings`} onClose={closeSettings}>
        <label className="schema-detail-field">Name{allowsEdit(key) ? <input aria-label={`${key} name`} value={port.name} disabled={!canEdit(key)}
          onChange={(event) => {
            trackRename('exec', key === 'execInPorts' ? 'input' : 'output', port.name, event.target.value);
            setExec(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
          }} /> : <output>{port.name}</output>}</label>
        <label className="schema-detail-field">Label<input aria-label={`${port.name} label`} value={port.label ?? ''} disabled={busy}
          onChange={(event) => setExec(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, label: event.target.value } : item))} />
        </label>
        <label className="schema-detail-field">Description<input aria-label={`${port.name} description`} value={port.description ?? ''} disabled={busy}
          onChange={(event) => setExec(key, (draft[key] ?? []).map((item, i) => i === index ? { ...item, description: event.target.value } : item))} /></label>
        {renderActions()}
        </SettingsDialog>}
      </div>)}
    </SchemaSection>)}
    <SchemaSection title="Commands" policy={policy('commands')} busy={busy} onAdd={() => {
      const commands = draft.commands ?? [];
      setCommands([...commands, { name: nextName(commands.map((command) => command.name), 'command'), definitionProtected: false, showOnNode: false }]);
    }}>
      {(draft.commands ?? []).map((command, index) => <div className="schema-item" key={index}>
        <div className="schema-item-main">{allowsEdit('commands') ? <input aria-label="Command name" value={command.name} disabled={!canEdit('commands')}
          onChange={(event) => {
            trackRename('command', null, command.name, event.target.value);
            setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
          }} /> : <span className="schema-item-name">{command.name}</span>}
          <VisibilityButton visible={command.showOnNode === true} busy={busy} onChange={(visible) =>
            setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, showOnNode: visible } : item))} />
          <button type="button" title={`Settings for ${command.name}`} aria-label={`Settings for ${command.name}`} onClick={() => setActiveSettings(`command:${index}`)}><Settings2 size={14} /></button>
          {allowsEdit('commands') && <button type="button" title={`Delete ${command.name}`} aria-label={`Delete ${command.name}`} disabled={!canDelete('commands', command.definitionProtected === true)}
            onClick={() => setCommands((draft.commands ?? []).filter((_, i) => i !== index))}><Trash2 size={13} /></button>}</div>
        {activeSettings === `command:${index}` && <SettingsDialog title={`${command.name} settings`} onClose={closeSettings}>
        <label className="schema-detail-field">Name{allowsEdit('commands') ? <input aria-label="Command name" value={command.name} disabled={!canEdit('commands')}
          onChange={(event) => {
            trackRename('command', null, command.name, event.target.value);
            setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, name: event.target.value } : item));
          }} /> : <output>{command.name}</output>}</label>
        <label className="schema-dialog-check"><input type="checkbox" checked={command.showOnNode === true} disabled={busy}
          onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, showOnNode: event.target.checked } : item))} />Show on node</label>
        <label className="schema-detail-field">Description<input aria-label={`${command.name} description`} value={command.description ?? ''} disabled={busy}
          onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? { ...item, description: event.target.value } : item))} /></label>
        {(allowsEdit('commands') || (command.params?.length ?? 0) > 0) && <div className="schema-params-heading"><span>Parameters</span>{allowsEdit('commands') && <button type="button" title={`Add parameter to ${command.name}`} aria-label={`Add parameter to ${command.name}`}
          disabled={!canEdit('commands')} onClick={() => {
            const params = command.params ?? [];
            setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: [...params, { name: nextName(params.map((param) => param.name), 'arg'), valueSchema: { type: 'string' }, valueRequired: false }],
            } : item));
          }}><Plus size={12} /></button>}</div>}
        {(command.params ?? []).map((param, paramIndex) => allowsEdit('commands') ? <div className="schema-item-main schema-param" key={paramIndex}>
          <input aria-label={`${command.name} parameter name`} value={param.name} disabled={!canEdit('commands')}
            onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: (item.params ?? []).map((value, j): CommandParamSpec => j === paramIndex ? { ...value, name: event.target.value } : value),
            } : item))} />
          <select className="schema-type-select" aria-label={`${param.name} parameter type`} value={valueType(param.valueSchema)} disabled={!canEdit('commands')}
            onChange={(event) => setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: (item.params ?? []).map((value, j) => j === paramIndex ? { ...value, valueSchema: { type: event.target.value } } : value),
            } : item))}>
            {['any', 'string', 'number', 'integer', 'boolean'].map((type) => <option key={type}>{type}</option>)}
          </select>
          <button type="button" title={`Delete ${param.name}`} aria-label={`Delete ${param.name}`} disabled={!canEdit('commands')}
            onClick={() => setCommands((draft.commands ?? []).map((item, i) => i === index ? {
              ...item, params: (item.params ?? []).filter((_, j) => j !== paramIndex),
            } : item))}><Trash2 size={12} /></button>
        </div> : <div className="schema-item-main schema-param" key={paramIndex}><span className="schema-item-name">{param.name}</span><span className="schema-kind">{valueType(param.valueSchema)}</span></div>)}
        {renderActions()}
        </SettingsDialog>}
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
    {activeSettings === null && renderActions()}
  </details>;
}
