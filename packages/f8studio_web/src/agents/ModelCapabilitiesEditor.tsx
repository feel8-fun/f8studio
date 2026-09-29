import { BrainCircuit, Image as ImageIcon } from 'lucide-react';
import type { AgentModelCapabilities } from '../api/contracts';

function capabilityLabel(value: boolean | null | undefined): string {
  return value === true ? 'Yes' : value === false ? 'No' : 'Unknown';
}

export function ModelCapabilitiesEditor({ model, capabilities, onChange }: {
  readonly model: string;
  readonly capabilities: readonly AgentModelCapabilities[];
  readonly onChange: (field: 'imageInput' | 'thinking', value: boolean | null) => void;
}) {
  if (!model) return null;
  const selected = capabilities.find((item) => item.modelId === model);
  const imageSource = selected?.imageSource ?? selected?.source;
  const thinkingSource = selected?.thinkingSource ?? selected?.source;
  const sourceLabel = (source: string | undefined, value: boolean | null | undefined) =>
    source === 'manual' ? 'manually confirmed' : source === 'legacy' ? 'previous setting' : value == null ? 'not reported by API' : 'reported by API';
  const parseValue = (value: string): boolean | null => value === 'yes' ? true : value === 'no' ? false : null;
  return <div className="agent-model-capabilities" aria-label="Model capabilities">
    <label className="agent-capability-control" title={`Image input: ${capabilityLabel(selected?.imageInput)} (${sourceLabel(imageSource, selected?.imageInput)})`}>
      <ImageIcon size={16} aria-hidden="true" /><span>Images</span>
      <select aria-label="Image input" value={selected?.imageSource === 'manual' ? selected.imageInput ? 'yes' : 'no' : 'unknown'} onChange={(event) => onChange('imageInput', parseValue(event.target.value))}>
        <option value="unknown">{selected?.imageSource === 'manual' ? 'Clear override' : `Auto: ${capabilityLabel(selected?.imageInput)}`}</option>
        <option value="yes">Yes</option><option value="no">No</option>
      </select>
    </label>
    <label className="agent-capability-control" title={`Reasoning: ${capabilityLabel(selected?.thinking)} (${sourceLabel(thinkingSource, selected?.thinking)})`}>
      <BrainCircuit size={16} aria-hidden="true" /><span>Reasoning</span>
      <select aria-label="Reasoning support" value={selected?.thinkingSource === 'manual' ? selected.thinking ? 'yes' : 'no' : 'unknown'} onChange={(event) => onChange('thinking', parseValue(event.target.value))}>
        <option value="unknown">{selected?.thinkingSource === 'manual' ? 'Clear override' : `Auto: ${capabilityLabel(selected?.thinking)}`}</option>
        <option value="yes">Yes</option><option value="no">No</option>
      </select>
    </label>
  </div>;
}

