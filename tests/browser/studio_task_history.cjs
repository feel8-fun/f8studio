// Run against the built Studio to verify history layout and browser persistence.
const assert = require('node:assert/strict');
const { existsSync } = require('node:fs');
const { chromium, expect } = require('../../extensions/f8webstudio/f8studio_web/node_modules/@playwright/test');
const baseUrl = process.env.F8_STUDIO_URL || 'http://127.0.0.1:8210';
const job = {
  jobId: 'history-fixture', extensionId: 'diagnostics', extensionVersion: '1', toolId: 'inspect',
  arguments: {}, status: 'succeeded', createdAt: '2026-10-07T12:00:00Z', updatedAt: '2026-10-07T12:00:01Z',
  result: { schemaVersion: 'f8toolResult/1', success: true, message: 'Completed', data: { payload: 'x'.repeat(20000) } },
  error: '', log: 'Detailed fixture log',
};
const task = {
  jobId: 'maintenance-fixture', request: { action: 'install-extension', extensionId: 'fixture', environmentId: null, package: null, location: null, sha256: null },
  state: 'succeeded', createdAt: 1, startedAt: 2, finishedAt: 3, detail: 'Completed fixture', cancellable: false, cancelRequested: false,
};
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.F8_CHROMIUM_EXECUTABLE || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined) });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/tool-jobs', route => route.fulfill({ json: [job] }));
  await page.route('**/api/extension-tools', route => route.fulfill({ json: [{ extensionId: 'diagnostics', toolId: 'inspect', name: 'Fixture inspect', description: '', requiresConfirmation: false, allowConcurrent: false, fields: [] }] }));
  await page.route('**/api/management-jobs', route => route.fulfill({ json: [task] }));
  try {
    for (const view of ['extensions', 'environments']) {
      await page.goto(`${baseUrl}/?view=${view}`);
      const trigger = page.getByRole('button', { name: 'Tasks · 0 active' });
      await expect(trigger).toBeVisible();
      await expect(page.locator('.services-body .management-tasks')).toHaveCount(0);
      await expect(page.getByRole('region', { name: 'Maintenance tasks' })).toHaveCount(0);
      const before = await page.locator('.services-body').boundingBox();
      await trigger.click();
      await expect(page.getByRole('region', { name: 'Maintenance tasks' })).toBeVisible();
      const after = await page.locator('.services-body').boundingBox();
      assert.deepEqual(after, before, 'Opening tasks must not push the page content');
      await page.getByRole('button', { name: 'Close tasks' }).click();
      await expect(page.getByRole('region', { name: 'Maintenance tasks' })).toHaveCount(0);
    }
    await page.goto(`${baseUrl}/?view=tools`);
    const history = page.locator('.tool-history');
    await expect(history.locator('summary').first()).toHaveText('Task history · 1 records');
    assert((await history.boundingBox()).height < 50, 'Collapsed history must occupy only one line');
    await expect(page.locator('.tool-result')).toBeHidden();
    await history.locator('summary').first().click();
    await expect(page.getByRole('button', { name: 'Delete record' })).toBeVisible();
    await expect(page.locator('.tool-result')).toBeHidden();
    await page.getByText('Result / logs', { exact: true }).click();
    await expect(page.locator('.tool-result')).toBeVisible();
    await history.locator('summary').first().click();
    await expect(page.locator('.tool-result')).toBeHidden();
    await history.locator('summary').first().click();
    await page.getByRole('button', { name: 'Delete record' }).click();
    await expect(page.locator('.tool-job')).toHaveCount(0);
    await page.reload();
    await expect(history.locator('summary').first()).toHaveText('Task history · 0 records');
    await history.locator('summary').first().click();
    await page.getByRole('button', { name: 'Close history' }).click();
    await page.reload();
    await expect(page.getByRole('button', { name: 'Show task history' })).toBeVisible();
    await expect(history).toHaveCount(0);
    assert.deepEqual(errors, []);
    console.log('PASS: built Studio toolbar tasks, stable content layout, one-line history, collapse, deletion and reload persistence');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
