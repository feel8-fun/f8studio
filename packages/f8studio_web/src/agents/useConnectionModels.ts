import { useState } from 'react';
import { probeAgentConnection } from '../api/client';
import type { AgentConnectionProbe, AgentModelCapabilities, AgentProviderSettings, CreateAgentConnection } from '../api/contracts';
import { mergeCapabilities } from './modelCapabilities';

interface ConnectionModelsOptions {
  readonly settings?: AgentProviderSettings;
  readonly protocol: CreateAgentConnection['protocol'] | null;
  readonly endpoint: string;
  readonly apiKey: string;
  readonly providerId?: string;
  readonly setBusy: (busy: boolean) => void;
  readonly setError: (error: string) => void;
}

/** Shared catalog refresh policy for both new and saved connections. */
export function useConnectionModels({ settings, protocol, endpoint, apiKey, providerId, setBusy, setError }: ConnectionModelsOptions) {
  const [model, setModel] = useState(settings?.model ?? '');
  const [models, setModels] = useState<readonly string[]>(settings?.models ?? (settings?.model ? [settings.model] : []));
  const [modelCapabilities, setModelCapabilities] = useState<readonly AgentModelCapabilities[]>(settings?.modelCapabilities ?? []);
  const [probe, setProbe] = useState<AgentConnectionProbe | null>(null);

  const test = async (verifyModel: boolean) => {
    if (protocol === null) return;
    setBusy(true); setError(''); setProbe(null);
    try {
      const result = await probeAgentConnection({
        providerId, protocol, endpoint: endpoint.trim(), apiKey: apiKey.trim(), model: model.trim(), verifyModel,
      });
      setProbe(result);
      if (!verifyModel && result.connected && result.verified === 'catalog') {
        setModels(result.models);
        setModelCapabilities((current) => mergeCapabilities(result.models, result.modelCapabilities ?? [], current));
        if (result.models.length > 0 && !result.models.includes(model.trim())) setModel(result.models[0] ?? '');
      }
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };

  return { model, setModel, models, setModels, modelCapabilities, setModelCapabilities, probe, setProbe, test };
}
