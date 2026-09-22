import { useEffect, useMemo, useState } from 'react';

import type { GraphNode, JsonValue, StateSpec } from '../api/contracts';

export function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isJsonValue);
  if (typeof value !== 'object') return false;
  return Object.values(value).every(isJsonValue);
}

function controlName(uiControl: string | undefined): string {
  return (uiControl ?? '').split('[', 1)[0]?.trim().toLowerCase() ?? '';
}

function poolField(uiControl: string | undefined): string | null {
  const match = /^(?:select|multiselect)\[([A-Za-z_][A-Za-z0-9_]*)\]$/.exec(uiControl?.trim() ?? '');
  return match?.[1] ?? null;
}

function fieldValue(node: GraphNode, fieldName: string): JsonValue {
  const field = (node.spec.stateFields ?? []).find((candidate) => candidate.name === fieldName);
  return node.stateValues[fieldName] ?? field?.valueSchema.default ?? null;
}

function numericBound(node: GraphNode, field: StateSpec, bound: 'minimum' | 'maximum'): number | undefined {
  const schemaBound = field.valueSchema[bound];
  if (typeof schemaBound === 'number') return schemaBound;
  const siblingName = bound === 'minimum' ? 'min' : 'max';
  const siblingValue = fieldValue(node, siblingName);
  return typeof siblingValue === 'number' ? siblingValue : undefined;
}

function displayValue(value: JsonValue): string {
  if (typeof value === 'string') return value;
  if (value === null) return 'null';
  return JSON.stringify(value);
}

export function StateFieldControl({
  node,
  field,
  disabled,
  connected = false,
  compact = false,
  onCommit,
}: {
  readonly node: GraphNode;
  readonly field: StateSpec;
  readonly disabled: boolean;
  readonly connected?: boolean;
  readonly compact?: boolean;
  readonly onCommit: (value: JsonValue) => void;
}) {
  const value = fieldValue(node, field.name);
  const [draft, setDraft] = useState(displayValue(value));
  useEffect(() => setDraft(displayValue(value)), [value]);
  const readOnly = field.access === 'ro';
  const controlDisabled = disabled || connected;
  const control = controlName(field.uiControl);
  const options = useMemo(() => {
    if (field.valueSchema.enum !== undefined) return field.valueSchema.enum;
    const pool = poolField(field.uiControl);
    const poolValue = pool === null ? null : fieldValue(node, pool);
    return Array.isArray(poolValue) ? poolValue.filter((item) =>
      typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') : [];
  }, [field.uiControl, field.valueSchema.enum, node]);
  const label = field.label ?? field.name;
  const title = `${label}${connected ? ' (driven by upstream state)' : ''}`;
  const shellClass = `${compact ? 'state-control state-control-inline nodrag nowheel' : 'state-control state-control-inspector'}${connected ? ' state-control-connected' : ''}`;
  const commitChanged = (nextValue: JsonValue): void => {
    if (readOnly || controlDisabled) return;
    if (JSON.stringify(nextValue) !== JSON.stringify(value)) onCommit(nextValue);
  };
  const commitJsonDraft = (text: string, target: HTMLInputElement | HTMLTextAreaElement): void => {
    if (controlDisabled) return;
    try {
      const parsed: unknown = JSON.parse(text);
      if (!isJsonValue(parsed)) throw new SyntaxError('Value must contain finite JSON values');
      target.setCustomValidity('');
      commitChanged(parsed);
    } catch (reason) {
      if (!(reason instanceof SyntaxError)) throw reason;
      target.setCustomValidity(reason.message);
      target.reportValidity();
    }
  };

  if (readOnly) {
    return <label className={`${shellClass} state-control-readonly`} title={title}>
      {!compact && <span>{label}</span>}
      <output>{displayValue(value)}</output>
      {connected && !compact && <small aria-hidden="true">Upstream</small>}
    </label>;
  }
  if (control === 'button') {
    const current = typeof value === 'number' ? value : 0;
    return <button className={`${shellClass} state-trigger-button`} type="button" title={title} disabled={controlDisabled} onClick={() => commitChanged(current + 1)}>
      {compact ? label : `Trigger ${label}`}
    </button>;
  }
  if (control === 'multiselect') {
    const selected = Array.isArray(value) ? value.filter((item) =>
      typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') : [];
    const choices = [...options];
    for (const item of selected) {
      if (!choices.some((choice) => JSON.stringify(choice) === JSON.stringify(item))) choices.push(item);
    }
    return <label className={shellClass} title={title}>
      {!compact && <span>{label}</span>}
      <select multiple disabled={controlDisabled} size={compact ? 1 : Math.min(Math.max(choices.length, 2), 5)} value={selected.map((item) => JSON.stringify(item))} onChange={(event) => {
        const parsed = [...event.currentTarget.selectedOptions].map((option): unknown => JSON.parse(option.value));
        if (parsed.every(isJsonValue)) commitChanged(parsed);
      }}>
        {choices.map((option) => <option key={JSON.stringify(option)} value={JSON.stringify(option)}>{String(option)}</option>)}
      </select>
      {connected && !compact && <small aria-hidden="true">Upstream</small>}
    </label>;
  }
  if (options.length > 0 || control === 'select') {
    return <label className={shellClass} title={title}>
      {!compact && <span>{label}</span>}
      <select disabled={controlDisabled} value={JSON.stringify(value)} onChange={(event) => {
        const parsed: unknown = JSON.parse(event.target.value);
        if (isJsonValue(parsed)) commitChanged(parsed);
      }}>
        {options.length === 0 && <option value={JSON.stringify(value)}>{displayValue(value)}</option>}
        {options.map((option) => <option key={JSON.stringify(option)} value={JSON.stringify(option)}>{String(option)}</option>)}
      </select>
      {connected && !compact && <small aria-hidden="true">Upstream</small>}
    </label>;
  }
  if (field.valueSchema.type === 'boolean' || control === 'toggle') {
    return <label className={`${shellClass} state-toggle`} title={title}>
      <input type="checkbox" checked={value === true} disabled={controlDisabled} onChange={(event) => commitChanged(event.target.checked)} />
      {!compact && <span>{label}</span>}
      {connected && !compact && <small aria-hidden="true">Upstream</small>}
    </label>;
  }
  if (field.valueSchema.type === 'number' || field.valueSchema.type === 'integer') {
    const minimum = numericBound(node, field, 'minimum');
    const maximum = numericBound(node, field, 'maximum');
    const step = field.valueSchema.multipleOf ?? (field.valueSchema.type === 'integer' ? 1 : 'any');
    if (control === 'slider' && minimum !== undefined && maximum !== undefined) {
      const numericValue = typeof value === 'number' ? value : minimum;
      return <label className={`${shellClass} state-slider`} title={title}>
        {!compact && <span>{label}</span>}
        <input type="range" min={minimum} max={maximum} step={step} value={draft} disabled={controlDisabled} onChange={(event) => setDraft(event.target.value)} onPointerUp={() => {
          const parsed = Number(draft);
          if (Number.isFinite(parsed)) commitChanged(parsed);
        }} onKeyUp={() => {
          const parsed = Number(draft);
          if (Number.isFinite(parsed)) commitChanged(parsed);
        }} />
        <output>{draft}</output>
        {connected && !compact && <small aria-hidden="true">Upstream</small>}
      </label>;
    }
    return <label className={shellClass} title={title}>
      {!compact && <span>{label}</span>}
      <input type="number" value={draft} min={minimum} max={maximum} step={step} readOnly={controlDisabled} onChange={(event) => setDraft(event.target.value)} onBlur={(event) => {
        const parsed = Number(event.target.value);
        if (!Number.isFinite(parsed)) return;
        if (field.valueSchema.type === 'integer' && !Number.isInteger(parsed)) return;
        if (minimum !== undefined && parsed < minimum) return;
        if (maximum !== undefined && parsed > maximum) return;
        commitChanged(parsed);
      }} />
      {connected && !compact && <small aria-hidden="true">Upstream</small>}
    </label>;
  }
  if (field.valueSchema.type === 'string') {
    const multiline = control === 'code' || control === 'wrapline';
    return <label className={`${shellClass} ${multiline ? 'state-text-code' : ''}`} title={title}>
      {!compact && <span>{label}</span>}
      {multiline && !compact
        ? <textarea rows={4} value={draft} readOnly={controlDisabled} onChange={(event) => setDraft(event.target.value)} onBlur={() => commitChanged(draft)} />
        : <input value={draft} readOnly={controlDisabled} onChange={(event) => setDraft(event.target.value)} onBlur={() => commitChanged(draft)} />}
      {connected && !compact && <small aria-hidden="true">Upstream</small>}
    </label>;
  }
  if (compact) {
    return <label className={`${shellClass} state-text-json`} title={title}>
      <input value={draft} readOnly={controlDisabled} onChange={(event) => setDraft(event.target.value)} onBlur={(event) => commitJsonDraft(event.target.value, event.target)} />
    </label>;
  }
  return <label className={shellClass} title={title}>
    {!compact && <span>{label}</span>}
    <textarea rows={4} value={draft} readOnly={controlDisabled} onChange={(event) => setDraft(event.target.value)} onBlur={(event) => commitJsonDraft(event.target.value, event.target)} />
    {connected && !compact && <small aria-hidden="true">Upstream</small>}
  </label>;
}
