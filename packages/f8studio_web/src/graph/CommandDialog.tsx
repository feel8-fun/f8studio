import { Play, X } from 'lucide-react';
import { useEffect, useState } from 'react';

import { invokeRuntimeCommand, setRuntimeState } from '../api/client';
import type { CommandParamSpec, CommandSpec, GraphNode, JsonValue } from '../api/contracts';

function initialValue(param: CommandParamSpec): string {
  const value = param.valueSchema.default;
  return value === undefined ? '' : typeof value === 'string' ? value : JSON.stringify(value);
}

function parseValue(param: CommandParamSpec, text: string): JsonValue | undefined {
  if (text.trim() === '') {
    if (param.required) throw new Error(`${param.name} is required`);
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

function rejectedMessage(value: JsonValue): string | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const result = value as Readonly<Record<string, JsonValue>>;
  if (result.success !== false) return null;
  return typeof result.errorMessage === 'string' ? result.errorMessage : 'Command rejected by runtime';
}

export function CommandDialog({ node, command, onClose }: {
  readonly node: GraphNode;
  readonly command: CommandSpec;
  readonly onClose: () => void;
}) {
  const [values, setValues] = useState<Readonly<Record<string, string>>>(() =>
    Object.fromEntries((command.params ?? []).map((param) => [param.name, initialValue(param)])));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<JsonValue | null>(null);

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
    setResult(null);
    try {
      const params: Record<string, JsonValue> = {};
      for (const param of command.params ?? []) {
        const value = parseValue(param, values[param.name] ?? '');
        if (value !== undefined) params[param.name] = value;
      }
      let response: JsonValue;
      if (node.kind === 'service') {
        response = await invokeRuntimeCommand(node.serviceId, command.name, params);
      } else {
        const port = node.ports.find((item) => item.kind === 'command' && item.direction === 'input' && item.name === command.name);
        if (port === undefined) throw new Error(`Command input ${command.name} is missing from ${node.name}`);
        response = await setRuntimeState(node.serviceId, node.nodeId, port.runtimeName, params);
      }
      const rejection = rejectedMessage(response);
      if (rejection !== null) throw new Error(rejection);
      setResult(node.kind === 'operator' ? { accepted: true } : response);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : 'Command failed');
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
        <span>{param.name}{param.required ? ' *' : ''}</span>
        {param.valueSchema.enum ? <select value={values[param.name] ?? ''} disabled={pending} onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))}>
          <option value="" disabled={param.required}>{param.required ? 'Select a value' : 'Unset'}</option>
          {param.valueSchema.enum.map((value) => <option key={JSON.stringify(value)} value={typeof value === 'string' ? value : JSON.stringify(value)}>{String(value)}</option>)}
        </select> : param.valueSchema.type === 'boolean' ? <select value={values[param.name] ?? ''} disabled={pending} onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))}>
          <option value="" disabled={param.required}>{param.required ? 'Select a value' : 'Unset'}</option><option value="true">True</option><option value="false">False</option>
        </select> : param.valueSchema.type === 'object' || param.valueSchema.type === 'array' ? <textarea
          value={values[param.name] ?? ''} disabled={pending} placeholder={param.valueSchema.type === 'array' ? '[]' : '{}'}
          onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))} /> : <input
          type={param.valueSchema.type === 'number' || param.valueSchema.type === 'integer' ? 'number' : 'text'}
          step={param.valueSchema.type === 'integer' ? 1 : 'any'} value={values[param.name] ?? ''} disabled={pending}
          onChange={(event) => setValues((current) => ({ ...current, [param.name]: event.target.value }))} />}
        {param.description && <small>{param.description}</small>}
      </label>)}</div>
      {error && <p className="error-text" role="alert">{error}</p>}
      {result !== null && <p className="command-result" role="status">{node.kind === 'operator' ? 'Submitted to runtime' : `Result: ${JSON.stringify(result)}`}</p>}
      <footer><button type="button" className="command-button primary" disabled={pending} onClick={() => void invoke()}><Play size={14} />{pending ? 'Running' : 'Run'}</button></footer>
    </section>
  </div>;
}
