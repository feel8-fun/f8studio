import { expect, test } from '@playwright/test';

test('adds a typed Decision node and exposes its separate provider settings', async ({ page, request }, testInfo) => {
  const projectId = `decision-e2e-${testInfo.project.name}-${Date.now()}`;
  expect((await request.post('/api/projects', { data: { projectId, name: 'Decision node test' } })).ok()).toBeTruthy();
  try {
    await page.goto('/');
    await page.locator('#project-select').selectOption(projectId);
    await page.getByLabel('Search nodes').fill('f8.pyengine');
    await page.getByRole('button', { name: /PyEngine/ }).click();
    await expect(page.locator('.flow-node-service')).toHaveCount(1);
    await page.getByLabel('Search nodes').fill('f8.decision');
    await page.getByRole('button', { name: /Decision f8\.decision/ }).click();
    await expect(page.locator('.flow-node-operator')).toHaveCount(1);
    const decision = page.locator('.flow-node-operator');
    await expect(decision).toContainText('decided');
    await expect(decision).toContainText('uncertain');
    await expect(decision).toContainText('probabilities');
    await expect(decision).toContainText('video');
    const response = await request.get(`/api/projects/${projectId}`);
    const project = await response.json() as { document: { nodes: { kind: string; stateValues: Record<string, unknown>; spec: { stateFields: { name: string; valueSchema: { default?: unknown } }[] } }[] } };
    const operator = project.document.nodes.find((node) => node.kind === 'operator');
    expect(operator?.spec.stateFields.find((field) => field.name === 'questions')?.valueSchema.default).toHaveProperty('decision');
    expect(operator?.stateValues).not.toHaveProperty('apiKey');
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Agent settings' });
    await dialog.getByRole('combobox', { name: 'Provider', exact: true }).selectOption('typesafe');
    await expect(dialog).toContainText('image input is not supported');
    await expect(dialog.getByLabel('Model ID')).toHaveValue('jev-latest');
    await dialog.getByRole('combobox', { name: 'Provider', exact: true }).selectOption('systemone_local');
    await expect(dialog.getByLabel('Host accepts JPEG images in state.image')).toBeVisible();
    await expect(dialog.getByLabel('API key', { exact: true })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('decision-provider.png') });
    const chatProviders = await (await request.get('/api/agents/providers')).json() as { providerId: string }[];
    expect(chatProviders.some((provider) => provider.providerId === 'typesafe')).toBe(false);
  } finally {
    expect((await request.delete(`/api/projects/${projectId}`)).status()).toBe(204);
  }
});
