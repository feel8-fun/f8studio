import { expect, test } from '@playwright/test';

test('local workspaces operate without Qt', async ({ page }, testInfo) => {
  const consoleErrors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Graph Editor' })).toBeVisible();

  await page.getByRole('button', { name: 'Assets' }).click();
  await expect(page.getByRole('heading', { name: 'Assets' })).toBeVisible();
  await page.getByRole('button', { name: 'Component', exact: true }).click();
  const assetName = `Workspace ${testInfo.project.name}`;
  await page.getByRole('textbox', { name: 'Asset name' }).fill(assetName);
  await page.getByRole('button', { name: 'Save version' }).click();
  await expect(page.getByText('Saved version 2')).toBeVisible();
  await expect(page.getByRole('button', { name: new RegExp(assetName) }).first()).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('assets-workspace.png'), fullPage: true });

  await page.keyboard.press('Control+4');
  await expect(page.getByRole('heading', { name: 'Code & Schema' })).toBeVisible();
  await expect(page.locator('.monaco-editor')).toBeVisible();
  await page.getByRole('button', { name: 'Analyze' }).click();
  await expect(page.getByText(/Valid.*basedpyright/)).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('code-workspace.png'), fullPage: true });

  await page.getByRole('button', { name: 'Outputs' }).click();
  await expect(page.getByRole('heading', { name: 'Live Outputs' })).toBeVisible();
  await expect(page.getByText('Event stream online')).toBeVisible();
  await page.getByRole('tab', { name: 'Template' }).click();
  await expect(page.getByRole('textbox', { name: 'Template match service id' })).toBeVisible();

  await page.getByRole('button', { name: 'Local integrations' }).click();
  await expect(page.getByRole('heading', { name: 'Local Integrations' })).toBeVisible();
  await expect(page.getByText('udp skeleton verify')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Skeleton UDP' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('local-integrations-workspace.png'), fullPage: true });

  await page.evaluate(async (createdName) => {
    const response = await fetch('/api/assets');
    const assets = await response.json() as { assetId: string; name: string }[];
    await Promise.all(assets.filter((asset) => asset.name === createdName).map((asset) =>
      fetch(`/api/assets/${encodeURIComponent(asset.assetId)}`, { method: 'DELETE' })));
  }, assetName);

  expect(consoleErrors).toEqual([]);
});

test('configures and restores a native global hotkey binding', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile', 'Global hotkey setup is verified in the desktop Inspector');
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();

  await page.evaluate(async () => {
    const response = await fetch('/api/local/hotkeys');
    const bindings = await response.json() as { bindingId: string; accelerator: string }[];
    await Promise.all(bindings.filter((binding) => binding.accelerator === 'Ctrl+Alt+P').map((binding) =>
      fetch(`/api/local/hotkeys/${encodeURIComponent(binding.bindingId)}`, { method: 'DELETE' })));
  });

  await page.getByLabel('Search nodes').fill('f8.value_stepper');
  await page.locator('.catalog-list button').filter({ hasText: 'f8.value_stepper' }).click();
  await page.locator('.flow-node-operator .node-drag-handle').click({ force: true, position: { x: 24, y: 12 } });

  const input = page.getByRole('textbox', { name: 'Increase global hotkey' });
  await expect(input).toBeVisible();
  await input.fill('ctrl + alt + p');
  await page.getByRole('button', { name: 'Save Increase global hotkey' }).click();
  await expect(input).toHaveValue('Ctrl+Alt+P');
  await expect.poll(async () => page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/local/hotkeys?project_id=${encodeURIComponent(selectedProjectId)}`);
    const bindings = await response.json() as { accelerator: string; nodeId: string; field: string; status: string }[];
    return bindings.find((binding) => binding.field === 'increaseTrigger');
  }, projectId)).toMatchObject({ accelerator: 'Ctrl+Alt+P', nodeId: expect.any(String), status: expect.stringMatching(/registered|disabled/) });

  await page.reload();
  await expect(page.locator('#project-select')).toHaveValue(projectId);
  await page.locator('.flow-node-operator .node-drag-handle').click({ force: true, position: { x: 24, y: 12 } });
  await expect(page.getByRole('textbox', { name: 'Increase global hotkey' })).toHaveValue('Ctrl+Alt+P');
  await page.getByRole('button', { name: 'Clear Increase global hotkey' }).click();
  await expect(page.getByRole('textbox', { name: 'Increase global hotkey' })).toHaveValue('');
});
