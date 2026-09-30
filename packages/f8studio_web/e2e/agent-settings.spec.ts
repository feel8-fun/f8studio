import { expect, test } from '@playwright/test';

test('configures a provider through Settings without displaying stored credentials', async ({ page }, testInfo) => {
  let settings = { providerId: 'openai', displayName: 'OpenAI', model: 'initial-model', endpoint: '', apiKeySet: false, requiresApiKey: true, configured: false, source: 'environment', kind: 'agent', inputModalities: ['text'], supportsImage: false };
  await page.route('**/api/agents/providers/settings', (route) => route.fulfill({ json: [settings] }));
  await page.route('**/api/agents/providers/openai/settings', async (route) => {
    const input = route.request().postDataJSON() as { model: string; endpoint: string; apiKey: string };
    expect(input.apiKey).toBe('test-key-not-a-real-credential');
    settings = { ...settings, model: input.model, endpoint: input.endpoint, apiKeySet: true, configured: true, source: 'saved' };
    await route.fulfill({ json: settings });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Agent settings' });
  await expect(dialog).toBeVisible();
  await dialog.getByLabel('Default model').selectOption('');
  await dialog.getByLabel('Custom model ID').fill('configured-model');
  await dialog.getByLabel('API key', { exact: true }).fill('test-key-not-a-real-credential');
  await dialog.getByRole('button', { name: 'Save provider' }).click();
  await expect(dialog.getByRole('status')).toContainText('Available for new agent sessions');
  await expect(dialog.getByLabel('API key', { exact: true })).toHaveValue('');
  await expect(dialog.getByLabel('API key', { exact: true })).toHaveAttribute('type', 'password');
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('agent-settings.png') });
  await dialog.getByRole('button', { name: 'Close agent settings' }).click();
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await expect(dialog.getByLabel('Default model')).toHaveValue('configured-model');
  await expect(dialog.getByLabel('API key', { exact: true })).toHaveValue('');
});

test('discovers models for a named connection on desktop and mobile', async ({ page }, testInfo) => {
  const settings = { providerId: 'openai', displayName: 'OpenAI', model: '', endpoint: '', apiKeySet: false, requiresApiKey: true, configured: false, source: 'environment', kind: 'agent', inputModalities: ['text'], supportsImage: false };
  await page.route('**/api/agents/providers/settings', (route) => route.fulfill({ json: [settings] }));
  await page.route('**/api/agents/connections/probe', (route) => route.fulfill({ json: {
    connected: true, models: ['model-a', 'model-b'], detail: 'API connected; found 2 models', verified: 'catalog',
  } }));
  await page.route(/\/api\/agents\/connections$/, async (route) => {
    const input = route.request().postDataJSON() as { apiKey: string; model: string; models: string[] };
    expect(input.apiKey).toBe('test-secret');
    expect(input.models).toEqual(['model-a', 'model-b']);
    await route.fulfill({ json: {
      ...settings, providerId: 'connection_test', displayName: 'Work API', protocol: 'openai_chat',
      endpoint: 'https://gateway.example/v1', model: input.model, models: input.models,
      apiKeySet: true, configured: true, custom: true, source: 'saved',
    } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Agent settings' });
  await dialog.getByRole('button', { name: 'New provider connection' }).click();
  await dialog.getByLabel('Connection name').fill('Work API');
  await dialog.getByLabel('API protocol').selectOption('openai_chat');
  await dialog.getByLabel('Base URL').fill('https://gateway.example/v1');
  await dialog.getByLabel('API key').fill('test-secret');
  await dialog.getByRole('button', { name: 'Detect models' }).click();
  await expect(dialog.getByText('API connected; found 2 models')).toBeVisible();
  await expect(dialog.getByLabel('Default model')).toHaveValue('model-a');
  await expect(dialog.getByLabel('Default model').locator('option')).toHaveCount(3);
  await dialog.getByLabel('Default model').selectOption('model-b');
  await expect(dialog.getByLabel('Default model')).toHaveValue('model-b');
  await dialog.getByRole('button', { name: 'Add connection' }).click();
  await expect(dialog.getByRole('combobox', { name: 'Provider', exact: true })).toHaveValue('connection_test');
  await expect(dialog.getByLabel('API key', { exact: true })).toHaveValue('');
  expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('named-connection.png') });
});

test('passes session reasoning effort to a run without calling the provider', async ({ page }, testInfo) => {
  await page.goto('/');
  const request = page.request;
  const projectId = `effort-ui-${testInfo.project.name}-${Date.now()}`;
  const project = await request.post('/api/projects', { data: { projectId, name: 'Reasoning UI check' } });
  expect(project.status()).toBe(201);
  const connection = await request.post('/api/agents/connections', { data: {
    displayName: 'Effort UI host', protocol: 'openai_chat', endpoint: 'http://127.0.0.1:11434/v1',
    model: 'reasoning-model', models: ['reasoning-model'],
    modelCapabilities: [{ modelId: 'reasoning-model', imageInput: false, thinking: null }],
  } });
  expect(connection.status()).toBe(201);
  const providerId = (await connection.json() as { providerId: string }).providerId;
  let sessionId = '';
  try {
    const created = await request.post('/api/agents/sessions', { data: {
      projectId, providerId, modelId: 'reasoning-model', title: 'Reasoning UI check',
    } });
    expect(created.status()).toBe(201);
    const session = await created.json() as { sessionId: string; status: string };
    sessionId = session.sessionId;
    await page.route(`**/api/agents/sessions/${sessionId}/runs`, async (route) => {
      expect(route.request().postDataJSON()).toMatchObject({ prompt: 'Inspect graph', reasoningEffort: 'high' });
      await route.fulfill({ json: { ...session, status: 'running' } });
    });
    await page.goto(`/?view=agent&project=${projectId}&session=${sessionId}`);
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Agent settings' });
    await dialog.getByRole('combobox', { name: 'Provider', exact: true }).selectOption(providerId);
    await expect(dialog.getByLabel('Reasoning support')).toHaveValue('unknown');
    await dialog.getByLabel('Reasoning support').selectOption('yes');
    await page.screenshot({ path: testInfo.outputPath('reasoning-setting.png') });
    await dialog.getByRole('button', { name: 'Save provider' }).click();
    await expect(dialog.getByRole('status')).toContainText('Saved.');
    const saved = await (await request.get('/api/agents/providers/settings')).json() as {
      providerId: string; modelCapabilities: { modelId: string; thinking: boolean | null; thinkingSource: string | null }[]
    }[];
    expect(saved.find((item) => item.providerId === providerId)?.modelCapabilities).toEqual(expect.arrayContaining([
      expect.objectContaining({ modelId: 'reasoning-model', thinking: true, thinkingSource: 'manual' }),
    ]));
    await dialog.getByRole('button', { name: 'Close agent settings' }).click();
    await page.reload();
    const effort = page.getByRole('combobox', { name: 'Reasoning effort' });
    await expect(effort).toHaveValue('auto');
    await effort.selectOption('high');
    await page.getByRole('textbox', { name: 'Agent prompt' }).fill('Inspect graph');
    await page.screenshot({ path: testInfo.outputPath('reasoning-effort.png') });
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page.locator('.agent-status')).toHaveText('running');
  } finally {
    if (sessionId) await request.delete(`/api/agents/sessions/${sessionId}`);
    await request.delete(`/api/agents/connections/${providerId}`);
    await request.delete(`/api/projects/${projectId}`);
  }
});
