import { expect, test } from '@playwright/test';

test('shows service and deployment logs below the compact title bar', async ({ page }, testInfo) => {
  const timestamp = '2026-09-23T15:00:00.000Z';
  await page.route('**/api/logs', async (route) => route.fulfill({ json: [
    {
      eventId: 'service-line', serverEpoch: 'epoch-1', sequence: 1, type: 'service.log',
      scope: 'service:capture-1', timestamp,
      payload: { serviceId: 'capture-1', line: 'Capture stream started' },
    },
    {
      eventId: 'deploy-error', serverEpoch: 'epoch-1', sequence: 2, type: 'deploy.finished',
      scope: 'project:project-1', timestamp,
      payload: { status: 'failed', sourceGraphRevision: 4, errorMessage: 'all services rejected the deployment',
        serviceResults: [{ serviceId: 'capture-1', success: false, errorMessage: 'endpoint not ready' }] },
    },
  ] }));
  await page.goto('/?view=logs');

  await expect(page.locator('.topbar').getByRole('heading', { name: 'Log Center' })).toBeVisible();
  await expect(page.getByText('Capture stream started')).toBeVisible();
  await expect(page.getByText('capture-1: endpoint not ready')).toBeVisible();
  const header = await page.locator('.topbar').boundingBox();
  const content = await page.locator('.logs-workspace').boundingBox();
  expect(header).not.toBeNull();
  expect(content).not.toBeNull();
  expect(content!.y).toBe(header!.y + header!.height);
  await page.screenshot({ path: testInfo.outputPath('log-center.png'), fullPage: true });

  await page.getByRole('combobox', { name: 'Log level' }).selectOption('error');
  await expect(page.getByText('Capture stream started')).toHaveCount(0);
  await expect(page.getByText('capture-1: endpoint not ready')).toBeVisible();
  await page.getByRole('textbox', { name: 'Search logs' }).fill('no-match');
  await expect(page.getByText('No matching logs')).toBeVisible();
});

test('pins and reorders live outputs in a persistent dashboard', async ({ page }) => {
  await page.route('**/api/presentation', async (route) => route.fulfill({ json: [
    { nodeId: 'viz-first', command: 'viz.text.show', payload: { value: 'First' }, tsMs: 1 },
    { nodeId: 'viz-second', command: 'viz.text.show', payload: { value: 'Second' }, tsMs: 1 },
  ] }));
  await page.goto('/');
  await page.getByRole('complementary', { name: 'Workspace navigation' }).getByRole('button', { name: 'Outputs' }).click();
  await expect(page.locator('.output-panel')).toHaveCount(2);
  await page.getByRole('button', { name: 'Pin viz-first' }).click();
  await page.getByRole('button', { name: 'Pin viz-second' }).click();
  await page.getByRole('tab', { name: 'Pinned' }).click();
  await page.getByRole('button', { name: 'Move viz-second up' }).click();
  await expect(page.locator('.output-panel header > span')).toHaveText(['viz-second', 'viz-first']);

  await page.reload();
  await page.getByRole('tab', { name: 'Pinned' }).click();
  await expect(page.locator('.output-panel header > span')).toHaveText(['viz-second', 'viz-first']);
});

test('renders a live 3D output in the dashboard', async ({ page }, testInfo) => {
  await page.route('**/api/presentation', async (route) => route.fulfill({ json: [{
    nodeId: 'viz-skeleton', command: 'viz.three_d.scene', tsMs: 1,
    payload: {
      tsMs: 1, worldUp: '+y', people: [{ name: 'Test', bbox: null, skeletonProtocol: 'test',
        skeletonEdges: [[0, 1]], nodes: [
          { index: 0, name: 'Root', pos: [0, 0, 0], rot: null },
          { index: 1, name: 'Head', pos: [0, 2, 0], rot: null },
        ] },
      ],
    },
  }] }));
  await page.goto('/?view=outputs');
  const preview = page.getByTestId('three-preview-viz-skeleton');
  await expect(preview.locator('canvas')).toBeVisible();
  await expect.poll(() => preview.locator('canvas').evaluate((canvas) => {
    const context = canvas.getContext('webgl2');
    if (context === null) return 0;
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    context.readPixels(0, 0, canvas.width, canvas.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let visible = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if ((pixels[index] ?? 0) + (pixels[index + 1] ?? 0) + (pixels[index + 2] ?? 0) > 0) visible += 1;
    }
    return visible;
  })).toBeGreaterThan(1_000);
  await page.screenshot({ path: testInfo.outputPath('three-dashboard.png'), fullPage: true });
});

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
