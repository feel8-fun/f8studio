import { expect, test } from '@playwright/test';

test('resizes and collapses quick logs without replacing the full log center', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile', 'Resize geometry is verified on desktop');
  await page.route('**/api/logs?*', async (route) => route.fulfill({ json: [{
    eventId: 'quick-log', serverEpoch: 'test', sequence: 1,
    type: 'service.log', scope: 'service:capture', timestamp: '2026-01-01T12:00:00Z',
    payload: { serviceId: 'capture', line: 'Capture ready' },
  }] }));
  await page.goto('/');
  const dock = page.getByRole('region', { name: 'Quick logs' });
  await expect(dock.getByText('Capture ready')).toBeVisible();
  await expect(dock.locator('.graph-log-header .logs-toolbar-inline')).toBeVisible();
  const header = await dock.locator('.graph-log-header').boundingBox();
  const list = await dock.locator('.logs-list').boundingBox();
  if (header === null || list === null) throw new Error('Quick log layout unavailable');
  expect(Math.abs(list.y - (header.y + header.height))).toBeLessThan(2);
  await dock.getByRole('combobox', { name: 'Log level' }).selectOption('error');
  await expect(dock.getByText('Capture ready')).toHaveCount(0);
  await dock.getByRole('combobox', { name: 'Log level' }).selectOption('all');
  await expect(dock.getByText('Capture ready')).toBeVisible();
  const before = await dock.boundingBox();
  const handle = await page.getByRole('separator', { name: 'Resize quick logs' }).boundingBox();
  if (before === null || handle === null) throw new Error('Quick log resize geometry unavailable');
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  await page.mouse.move(handle.x + handle.width / 2, handle.y - 90, { steps: 5 });
  await page.mouse.up();
  const resized = await dock.boundingBox();
  expect(resized).not.toBeNull();
  expect(resized!.height).toBeGreaterThan(before.height + 70);

  await page.reload();
  const restored = await dock.boundingBox();
  expect(restored).not.toBeNull();
  expect(Math.abs(restored!.height - resized!.height)).toBeLessThan(2);
  await dock.getByRole('button', { name: 'Collapse quick logs' }).click();
  await expect(dock.getByRole('button', { name: 'Expand quick logs' })).toBeVisible();
  await expect(dock.getByRole('log')).toHaveCount(0);
  const collapsed = await dock.boundingBox();
  expect(collapsed!.height).toBeLessThan(35);
  await dock.getByRole('button', { name: 'Expand quick logs' }).click();
  await expect(dock.getByText('Capture ready')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('quick-logs-graph.png'), fullPage: true });

  const collapseHandle = await page.getByRole('separator', { name: 'Resize quick logs' }).boundingBox();
  const expanded = await dock.boundingBox();
  if (collapseHandle === null || expanded === null) throw new Error('Quick log collapse geometry unavailable');
  await page.mouse.move(collapseHandle.x + collapseHandle.width / 2, collapseHandle.y + collapseHandle.height / 2);
  await page.mouse.down();
  await page.mouse.move(collapseHandle.x + collapseHandle.width / 2, expanded.y + expanded.height - 4, { steps: 5 });
  await page.mouse.up();
  await expect(dock.getByRole('button', { name: 'Expand quick logs' })).toBeVisible();

  await dock.getByRole('button', { name: 'Open log center' }).click();
  await expect(page.getByRole('heading', { name: 'Log Center' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Log center' }).getByText('Capture ready')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('quick-logs-full-page.png'), fullPage: true });
});

test('fits quick log controls into the mobile title bar', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile', 'Mobile layout check');
  await page.route('**/api/logs?*', async (route) => route.fulfill({ json: [{
    eventId: 'mobile-log', serverEpoch: 'test', sequence: 1,
    type: 'service.log', scope: 'service:capture', timestamp: '2026-01-01T12:00:00Z',
    payload: { serviceId: 'capture', line: 'Capture ready' },
  }] }));
  await page.goto('/');
  const dock = page.getByRole('region', { name: 'Quick logs' });
  await expect(dock.getByText('Capture ready')).toBeVisible();
  await expect(dock.locator('.graph-log-header .logs-toolbar-inline')).toBeVisible();
  const geometry = await page.evaluate(() => {
    const header = document.querySelector<HTMLElement>('.graph-log-header');
    const open = document.querySelector<HTMLElement>('.graph-log-open');
    if (header === null || open === null) throw new Error('Quick log header unavailable');
    return {
      pageFits: document.documentElement.scrollWidth <= window.innerWidth,
      controlsFit: open.getBoundingClientRect().right <= header.getBoundingClientRect().right + 1,
    };
  });
  expect(geometry).toEqual({ pageFits: true, controlsFit: true });
  await page.screenshot({ path: testInfo.outputPath('quick-logs-mobile.png'), fullPage: true });
});

test('keeps the log center inside the viewport and pages older entries on demand', async ({ page }, testInfo) => {
  await page.route('**/api/logs?*', async (route) => {
    const url = new URL(route.request().url());
    const before = Number(url.searchParams.get('before_sequence') ?? 501);
    const limit = Number(url.searchParams.get('limit') ?? 100);
    const start = Math.max(1, before - limit);
    const events = Array.from({ length: before - start }, (_, index) => {
      const sequence = start + index;
      return {
        eventId: `log-${sequence}`, serverEpoch: 'test', sequence,
        type: 'service.log', scope: 'service:player', timestamp: '2026-01-01T12:00:00Z',
        payload: { serviceId: 'player', line: `Frame ${sequence}: ${'decoded sample '.repeat(8)}` },
      };
    });
    await route.fulfill({ json: events });
  });

  await page.goto('/?view=logs');
  const rows = page.locator('.logs-row');
  await expect(rows).toHaveCount(100);
  const geometry = await page.evaluate(() => {
    const list = document.querySelector<HTMLElement>('.logs-list');
    if (list === null) throw new Error('Missing log list');
    return {
      pageFits: document.documentElement.scrollHeight <= window.innerHeight,
      listScrolls: list.scrollHeight > list.clientHeight,
    };
  });
  expect(geometry).toEqual({ pageFits: true, listScrolls: true });
  await expect(page.getByRole('button', { name: 'Assets' })).toBeVisible();

  await page.getByRole('button', { name: 'Load older' }).click();
  await expect(rows).toHaveCount(200);
  await expect(page.getByRole('button', { name: 'Latest' })).toBeVisible();
  await page.getByRole('button', { name: 'Load older' }).click();
  await expect(rows).toHaveCount(300);
  await page.getByRole('button', { name: 'Load older' }).click();
  await expect(rows).toHaveCount(300);
  await expect(rows.first()).toContainText('Frame 101:');
  await page.getByRole('button', { name: 'Latest' }).click();
  await expect(rows).toHaveCount(100);
  await expect(rows.first()).toContainText('Frame 401:');
  await page.screenshot({ path: testInfo.outputPath('bounded-log-center.png'), fullPage: true });
});
