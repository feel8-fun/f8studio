import { Play, X } from 'lucide-react';
import { useEffect, useState } from 'react';

import type { CommandParamSpec, CommandSpec, GraphNode, JsonValue } from '../api/contracts';
import { commandResultDetail, runCommand } from './runCommand';

function initialValue(param: CommandParamSpec): string {
  const value = param.valueSchema.default;
  return value === undefined ? '' : typeof value === 'string' ? value : JSON.stringify(value);
}

function parseValue(param: CommandParamSpec, text: string): JsonValue | undefined {
  if (text.trim() === '') {
    if (param.valueRequired) throw new Error(`${param.name} is required`);
    return undefined;
  }
  const type = param.valueSchema.type;
  if (type === 'string') return text;
  if (type === 'boolean') {
    if (text === 'true') return true;
    if (text === 'false') return false;
    throw new Error(`${param.name} must be true or false`);
  }
  if (type === 'number' || type === 'integer') {
    const value = Number(text);
    if (!Number.isFinite(value) || (type === 'integer' && !Number.isInteger(value))) throw new Error(`${param.name} must be a ${type}`);
    if (param.valueSchema.minimum !== undefined && value < param.valueSchema.minimum) throw new Error(`${param.name} must be at least ${param.valueSchema.minimum}`);
    if (param.valueSchema.maximum !== undefined && value > param.valueSchema.maximum) throw new Error(`${param.name} must be at most ${param.valueSchema.maximum}`);
    return value;
  }
  try {
    return JSON.parse(text) as JsonValue;
  } catch {
    throw new Error(`${param.name} must contain valid JSON`);
  }
}

export function CommandDialog({ node, command, onClose, onResult }: {
  readonly node: GraphNode;
  readonly command: CommandSpec;
  readonly onClose: () => void;
  readonly onResult: (kind: 'success' | 'error', title: string, detail: string) => void;
}) {
  const [values, setValues] = useState<Readonly<Record<string, string>>>(() =>
    Object.fromEntries((command.params ?? []).map((param) => [param.name, initialValue(param)])));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !pending) onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, pending]);

  const invoke = async () => {
    setPending(true);
    setError(null);
    let submitted = false;
    try {
      const params: Record<string, JsonValue> = {};
      for (const param of command.params ?? []) {
        const value = parseValue(param, values[param.name] ?? '');
        if (value !== undefined) params[param.name] = value;
      }
      submitted = true;
      const response = await runCommand(node, command, params);
      onResult('success', `${node.name}: ${command.name}`, commandResultDetail(node, response));
      onClose();
    } catch (reason: unknown) {
      const message = reason instanceof Error ? reason.message : 'Command failed';
      setError(message);
      if (submitted) onResult('error', `${node.name}: ${command.name} failed`, message);
    } finally {
      setPending(false);
    }
  };

  return <div className="command-dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !pending) onClose(); }}>
    <section className="command-dialog" role="dialog" aria-modal="true" aria-label={`Run ${command.name}`}>
      <header><div><strong>{command.name}</strong><span>{node.name}</span></div>
        <button type="button" className="icon-button" title="Close" aria-label="Close command" disabled={pending} onClick={onClose}><X size={16} /></button>
      </header>
      {command.description && <p>{command.description}</p>}
      <div className="command-params">{(command.params ?? []).map((param) => <label key={param.name} className="inspector-field">
        <span>{param.name}{param.valueRequired ? ' *' : ''}</span>
        {param.valueSchema.enum ? <select value={values[param.name] ?? ''} disabled={pending} onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))}>
          <option value="" disabled={param.valueRequired}>{param.valueRequired ? 'Select a value' : 'Unset'}</option>
          {param.valueSchema.enum.map((value) => <option key={JSON.stringify(value)} value={typeof value === 'string' ? value : JSON.stringify(value)}>{String(value)}</option>)}
        </select> : param.valueSchema.type === 'boolean' ? <select value={values[param.name] ?? ''} disabled={pending} onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))}>
          <option value="" disabled={param.valueRequired}>{param.valueRequired ? 'Select a value' : 'Unset'}</option><option value="true">True</option><option value="false">False</option>
        </select> : param.valueSchema.type === 'object' || param.valueSchema.type === 'array' ? <textarea
          value={values[param.name] ?? ''} disabled={pending} placeholder={param.valueSchema.type === 'array' ? '[]' : '{}'}
          onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))} /> : <input
          type={param.valueSchema.type === 'number' || param.valueSchema.type === 'integer' ? 'number' : 'text'}
          step={param.valueSchema.type === 'integer' ? 1 : 'any'} value={values[param.name] ?? ''} disabled={pending}
          onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))} />}
        {param.description && <small>{param.description}</small>}
      </label>)}</div>
      {error && <p className="error-text" role="alert">{error}</p>}
      <footer><button type="button" className="command-button primary" disabled={pending} onClick={() => void invoke()}><Play size={14} />{pending ? 'Running' : 'Run'}</button></footer>
    </section>
  </div>;
}
