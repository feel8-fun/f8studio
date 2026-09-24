import { expect, test } from '@playwright/test';

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
