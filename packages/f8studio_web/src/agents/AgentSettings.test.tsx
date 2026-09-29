import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { AGENT_PROVIDERS_CHANGED, AgentSettingsButton } from './AgentSettings';

const api = vi.hoisted(() => ({
  fetchAgentProviderSettings: vi.fn(), saveAgentProviderSettings: vi.fn(),
  createAgentConnection: vi.fn(), deleteAgentConnection: vi.fn(), probeAgentConnection: vi.fn(),
}));
vi.mock('../api/client', () => api);

const settings = { providerId: 'openai', displayName: 'OpenAI', model: 'example-model', endpoint: '', apiKeySet: true, requiresApiKey: true, configured: true, source: 'environment', kind: 'agent', inputModalities: ['text'], supportsImage: false };

beforeEach(() => {
  api.fetchAgentProviderSettings.mockResolvedValue([
    settings,
    { providerId: 'ollama', displayName: 'Ollama (local)', model: '', endpoint: 'http://localhost:11434/v1', apiKeySet: false, requiresApiKey: false, configured: false, source: 'environment', kind: 'agent', inputModalities: ['text'], supportsImage: false },
  ]);
  api.saveAgentProviderSettings.mockResolvedValue({ ...settings, source: 'saved' });
  api.probeAgentConnection.mockResolvedValue({ connected: true, models: ['vision-a', 'vision-b'], detail: 'API connected; found 2 models', verified: 'catalog' });
});
afterEach(() => { cleanup(); vi.resetAllMocks(); });

test('keeps an existing key when blank and notifies the provider selector after saving', async () => {
  const changed = vi.fn();
  window.addEventListener(AGENT_PROVIDERS_CHANGED, changed);
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  expect(await screen.findByLabelText('API key')).toHaveValue('');
  fireEvent.change(screen.getByLabelText('Default model'), { target: { value: '' } });
  fireEvent.change(screen.getByLabelText('Custom model ID'), { target: { value: 'new-model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('openai', expect.objectContaining({ model: 'new-model', models: ['example-model', 'new-model'], endpoint: '', apiKey: undefined, clearApiKey: false })));
  expect(await screen.findByRole('status')).toHaveTextContent('Saved.');
  expect(changed).toHaveBeenCalledOnce();
  window.removeEventListener(AGENT_PROVIDERS_CHANGED, changed);
});

test('clears the typed credential after saving and supports explicitly removing it', async () => {
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  const input = await screen.findByLabelText('API key');
  expect(input).toHaveAttribute('type', 'password');
  fireEvent.change(input, { target: { value: 'new-private-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(input).toHaveValue(''));
  expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('openai', expect.objectContaining({ apiKey: 'new-private-key' }));
  fireEvent.click(screen.getByLabelText('Remove saved API key'));
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('openai', expect.objectContaining({ clearApiKey: true, apiKey: undefined })));
});

test('deletes a legacy OpenAI connection from settings', async () => {
  api.deleteAgentConnection.mockResolvedValue(undefined);
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  expect(await screen.findByText('OpenAI', { selector: 'h3' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Delete connection' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
  await waitFor(() => expect(api.deleteAgentConnection).toHaveBeenCalledWith('openai'));
  expect(screen.queryByRole('option', { name: /OpenAI/ })).not.toBeInTheDocument();
});

test('configures a local provider without requiring a key', async () => {
  render(<AgentSettingsButton compact />);
  fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
  fireEvent.change(await screen.findByLabelText('Provider'), { target: { value: 'ollama' } });
  expect(screen.queryByLabelText('API key')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Default model'), { target: { value: 'local-model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('ollama', expect.objectContaining({ model: 'local-model', models: ['local-model'], endpoint: 'http://localhost:11434/v1', apiKey: undefined, clearApiKey: false })));
});

test('reports a save failure and leaves the draft available for retry', async () => {
  api.saveAgentProviderSettings.mockRejectedValue(new Error('Server unavailable'));
  render(<AgentSettingsButton />);
  screen.getByRole('button', { name: 'Configure agent providers' }).focus();
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  fireEvent.change(await screen.findByLabelText('Default model'), { target: { value: '' } });
  fireEvent.change(screen.getByLabelText('Custom model ID'), { target: { value: 'draft-model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Server unavailable');
  expect(screen.getByLabelText('Custom model ID')).toHaveValue('draft-model');
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Configure agent providers' })).toHaveFocus();
});

test('describes Jev as a text decision provider and saves it for graph nodes', async () => {
  const jev = { providerId: 'typesafe', displayName: 'TypeSafe Jev (decisions)', model: 'jev-latest', endpoint: 'https://api.typesafe.ai/v1', apiKeySet: false, requiresApiKey: true, configured: false, source: 'environment', kind: 'decision', inputModalities: ['text'], supportsImage: false };
  api.fetchAgentProviderSettings.mockResolvedValue([jev]);
  api.saveAgentProviderSettings.mockResolvedValue({ ...jev, apiKeySet: true, configured: true, source: 'saved' });
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  expect(await screen.findByText(/image input is not supported/)).toBeInTheDocument();
  expect(screen.getByLabelText('Decision node provider ID')).toHaveValue('typesafe');
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'jev-test-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  expect(await screen.findByRole('status')).toHaveTextContent('Available to Decision nodes');
  expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('typesafe', expect.objectContaining({ model: 'jev-latest', endpoint: 'https://api.typesafe.ai/v1', apiKey: 'jev-test-key' }));
});

test('discovers models and adds a named connection using a separate key', async () => {
  api.createAgentConnection.mockResolvedValue({
    ...settings, providerId: 'connection_123', displayName: 'Work gateway', protocol: 'openai_chat',
    endpoint: 'https://gateway.example/v1', model: 'vision-a', models: ['vision-a', 'vision-b'], custom: true,
  });
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  fireEvent.click(await screen.findByRole('button', { name: 'New provider connection' }));
  fireEvent.change(screen.getByLabelText('Connection name'), { target: { value: 'Work gateway' } });
  fireEvent.change(screen.getByLabelText('API protocol'), { target: { value: 'openai_chat' } });
  fireEvent.change(screen.getByLabelText('Base URL'), { target: { value: 'https://gateway.example/v1' } });
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'separate-secret' } });
  fireEvent.click(screen.getByRole('button', { name: 'Detect models' }));
  expect(await screen.findByText('API connected; found 2 models')).toBeInTheDocument();
  expect(screen.getByLabelText('Default model')).toHaveValue('vision-a');
  fireEvent.click(screen.getByRole('button', { name: 'Add connection' }));
  await waitFor(() => expect(api.createAgentConnection).toHaveBeenCalledWith(expect.objectContaining({
    displayName: 'Work gateway', protocol: 'openai_chat', apiKey: 'separate-secret',
    model: 'vision-a', models: ['vision-a', 'vision-b'],
  })));
  expect(screen.getByLabelText('Provider')).toHaveValue('connection_123');
  expect(screen.getByLabelText('API key')).toHaveValue('');
});

test('refreshes model metadata and lets the default model change after saving', async () => {
  api.fetchAgentProviderSettings.mockResolvedValue([{
    ...settings, providerId: 'connection_work', displayName: 'Work', custom: true,
    protocol: 'openai_chat', endpoint: 'https://gateway.example/v1', models: ['old-model'], model: 'old-model',
  }]);
  api.probeAgentConnection.mockResolvedValue({
    connected: true, models: ['text-model', 'vision-model'], detail: 'API connected; found 2 models', verified: 'catalog',
    modelCapabilities: [
      { modelId: 'text-model', imageInput: false, thinking: null },
      { modelId: 'vision-model', imageInput: true, thinking: true },
    ],
  });
  api.saveAgentProviderSettings.mockImplementation(async (_id, input) => ({
    ...settings, providerId: 'connection_work', displayName: 'Work', custom: true,
    protocol: 'openai_chat', endpoint: input.endpoint, model: input.model,
    models: input.models, modelCapabilities: input.modelCapabilities,
  }));
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Refresh models' }));
  await screen.findByText('API connected; found 2 models');
  expect(screen.getByLabelText('Default model')).toHaveValue('text-model');
  fireEvent.change(screen.getByLabelText('Default model'), { target: { value: 'vision-model' } });
  expect(screen.getByTitle(/Image input: Yes/)).toBeInTheDocument();
  expect(screen.getByTitle(/Reasoning: Yes/)).toBeInTheDocument();
  expect(screen.getByLabelText('Reasoning support')).toHaveValue('unknown');
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('connection_work', expect.objectContaining({
    model: 'vision-model', models: ['text-model', 'vision-model'], modelCapabilities: expect.arrayContaining([
      expect.objectContaining({ modelId: 'vision-model', imageInput: true, thinking: true, imageSource: 'catalog' }),
    ]),
  })));
});

test('allows a per-model reasoning confirmation when the API reports Unknown', async () => {
  api.fetchAgentProviderSettings.mockResolvedValue([{
    ...settings, providerId: 'connection_work', displayName: 'Work', custom: true,
    protocol: 'openai_chat', endpoint: 'https://gateway.example/v1', model: 'model-a',
    models: ['model-a', 'model-b'], modelCapabilities: [
      { modelId: 'model-a', imageInput: null, thinking: null },
      { modelId: 'model-b', imageInput: null, thinking: null },
    ],
  }]);
  api.probeAgentConnection.mockResolvedValue({
    connected: true, models: ['model-a', 'model-b'], detail: 'API connected; found 2 models', verified: 'catalog',
    modelCapabilities: [
      { modelId: 'model-a', imageInput: null, thinking: null },
      { modelId: 'model-b', imageInput: null, thinking: null },
    ],
  });
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  const reasoning = await screen.findByLabelText('Reasoning support');
  expect(reasoning).toBeEnabled();
  expect(reasoning).toHaveValue('unknown');
  fireEvent.change(reasoning, { target: { value: 'yes' } });
  expect(screen.getByTitle(/Reasoning: Yes \(manually confirmed\)/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Default model'), { target: { value: 'model-b' } });
  expect(reasoning).toHaveValue('unknown');
  fireEvent.change(reasoning, { target: { value: 'no' } });
  fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  await screen.findByText('API connected; found 2 models');
  expect(reasoning).toHaveValue('no');
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('connection_work', expect.objectContaining({
    modelCapabilities: expect.arrayContaining([
      expect.objectContaining({ modelId: 'model-a', thinking: true, thinkingSource: 'manual' }),
      expect.objectContaining({ modelId: 'model-b', thinking: false, thinkingSource: 'manual' }),
    ]),
  })));
});

test('keeps a model-specific image confirmation when refreshing an opaque catalog', async () => {
  api.fetchAgentProviderSettings.mockResolvedValue([{
    ...settings, providerId: 'connection_work', displayName: 'Work', custom: true,
    protocol: 'openai_chat', endpoint: 'https://gateway.example/v1', model: 'vision-model',
    models: ['vision-model', 'text-model'], modelCapabilities: [
      { modelId: 'vision-model', imageInput: true, thinking: null, imageSource: 'manual' },
      { modelId: 'text-model', imageInput: null, thinking: null },
    ],
  }]);
  api.probeAgentConnection.mockResolvedValue({
    connected: true, models: ['vision-model', 'text-model'], detail: 'API connected; found 2 models', verified: 'catalog',
    modelCapabilities: [
      { modelId: 'vision-model', imageInput: null, thinking: true },
      { modelId: 'text-model', imageInput: null, thinking: null },
    ],
  });
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  expect(await screen.findByTitle(/Image input: Yes \(manually confirmed\)/)).toBeInTheDocument();
  expect(screen.getByTitle(/Reasoning: Unknown/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  await screen.findByText('API connected; found 2 models');
  expect(screen.getByTitle(/Image input: Yes \(manually confirmed\)/)).toBeInTheDocument();
  expect(screen.getByTitle(/Reasoning: Yes \(reported by API\)/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Default model'), { target: { value: 'text-model' } });
  fireEvent.change(screen.getByLabelText('Image input'), { target: { value: 'no' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('connection_work', expect.objectContaining({
    modelCapabilities: expect.arrayContaining([
      expect.objectContaining({ modelId: 'vision-model', imageInput: true, imageSource: 'manual', thinking: true }),
      expect.objectContaining({ modelId: 'text-model', imageInput: false, imageSource: 'manual' }),
    ]),
  })));
});

test('testing one model keeps the connection model list', async () => {
  api.fetchAgentProviderSettings.mockResolvedValue([{
    ...settings, providerId: 'connection_work', displayName: 'Work', custom: true,
    protocol: 'openai_chat', endpoint: 'https://gateway.example/v1',
    models: ['model-a', 'model-b'], model: 'model-a',
  }]);
  api.probeAgentConnection.mockResolvedValue({
    connected: true, models: ['model-a'], detail: 'Model inference succeeded', verified: 'model',
  });
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Test model' }));
  await screen.findByText('Model inference succeeded');
  fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
  await waitFor(() => expect(api.saveAgentProviderSettings).toHaveBeenCalledWith('connection_work', expect.objectContaining({
    model: 'model-a', models: ['model-a', 'model-b'],
  })));
});

test('shows every saved model in the default model dropdown', async () => {
  const models = Array.from({ length: 27 }, (_, index) => `model-${index + 1}`);
  api.fetchAgentProviderSettings.mockResolvedValue([{
    ...settings, providerId: 'connection_many', displayName: 'Many models', custom: true,
    protocol: 'openai_responses', endpoint: 'https://gateway.example/v1', model: models[0], models,
  }]);
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  const picker = await screen.findByRole('combobox', { name: 'Default model' });
  expect(picker.querySelectorAll('option')).toHaveLength(28);
  fireEvent.change(picker, { target: { value: 'model-27' } });
  expect(picker).toHaveValue('model-27');
});

test('new connections replace removed models on refresh and invalidate probe status when edited', async () => {
  api.fetchAgentProviderSettings.mockResolvedValue([]);
  render(<AgentSettingsButton />);
  fireEvent.click(screen.getByRole('button', { name: 'Configure agent providers' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Detect models' }));
  await screen.findByText('API connected; found 2 models');
  expect(screen.getByLabelText('Default model')).toHaveValue('vision-a');
  api.probeAgentConnection.mockResolvedValue({
    connected: true, models: ['replacement'], detail: 'Updated catalog', verified: 'catalog',
  });
  fireEvent.click(screen.getByRole('button', { name: 'Detect models' }));
  await screen.findByText('Updated catalog');
  expect(screen.getByLabelText('Default model')).toHaveValue('replacement');
  fireEvent.change(screen.getByLabelText('Base URL'), { target: { value: 'https://other.example/v1' } });
  expect(screen.queryByText('Updated catalog')).not.toBeInTheDocument();
});
