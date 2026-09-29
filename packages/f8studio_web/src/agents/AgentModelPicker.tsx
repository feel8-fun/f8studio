import type { KeyboardEvent } from 'react';

interface AgentModelPickerProps {
  readonly label: string;
  readonly models: readonly string[];
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly disabled?: boolean;
  readonly compact?: boolean;
  readonly onKeyDown?: (event: KeyboardEvent<HTMLInputElement>) => void;
}

export function AgentModelPicker({ label, models, value, onChange, disabled = false, compact = false, onKeyDown }: AgentModelPickerProps) {
  const listed = models.includes(value);
  const manual = models.length === 0 || !listed;
  return <div className={`agent-model-picker${compact ? ' compact' : ''}`}>
    {models.length > 0 && <label>{compact ? null : label}
      <select aria-label={label} value={listed ? value : ''} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {models.map((model) => <option key={model} value={model}>{model}</option>)}
        <option value="">Enter model ID…</option>
      </select>
    </label>}
    {manual && <label>{compact ? null : models.length > 0 ? 'Custom model ID' : label}
      <input aria-label={models.length > 0 ? 'Custom model ID' : label} value={value} disabled={disabled} required={!compact} onChange={(event) => onChange(event.target.value)} onKeyDown={onKeyDown} placeholder="Model ID" autoComplete="off" />
    </label>}
  </div>;
}
