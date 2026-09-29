import type { AgentModelCapabilities } from '../api/contracts';

export function mergeCapabilities(models: readonly string[], detected: readonly AgentModelCapabilities[], current: readonly AgentModelCapabilities[]): AgentModelCapabilities[] {
  return models.map((modelId) => {
    const reported = detected.find((item) => item.modelId === modelId) ?? { modelId, imageInput: null, thinking: null };
    const previous = current.find((item) => item.modelId === modelId);
    const keepImage = previous?.imageSource === 'manual' ||
      ((previous?.imageSource ?? previous?.source) === 'legacy' && reported.imageInput == null);
    const keepThinking = previous?.thinkingSource === 'manual';
    return {
      ...reported,
      imageInput: keepImage ? (previous?.imageInput ?? null) : reported.imageInput,
      thinking: keepThinking ? (previous?.thinking ?? null) : reported.thinking,
      imageSource: keepImage ? (previous?.imageSource ?? 'legacy') : 'catalog',
      thinkingSource: keepThinking ? 'manual' : 'catalog',
    };
  });
}

export function setCapability(capabilities: readonly AgentModelCapabilities[], modelId: string, field: 'imageInput' | 'thinking', value: boolean | null): AgentModelCapabilities[] {
  const current = capabilities.find((item) => item.modelId === modelId) ?? { modelId, imageInput: null, thinking: null };
  const changed: AgentModelCapabilities = field === 'imageInput'
    ? { ...current, imageInput: value, imageSource: value === null ? 'catalog' : 'manual' }
    : { ...current, thinking: value, thinkingSource: value === null ? 'catalog' : 'manual' };
  return [...capabilities.filter((item) => item.modelId !== modelId), changed];
}

