import { expect, test } from '@playwright/test';

test('agent builds a graph with exact approvals and updates an open graph viewer', async ({ context, page, request }, testInfo) => {
  const suffix = `${testInfo.project.name}-${Date.now()}`.replaceAll(/[^a-zA-Z0-9_-]/g, '-');
  const projectId = `agent-e2e-${suffix}`;
  const created = await request.post('/api/projects', {
    data: { projectId, name: `Agent E2E ${suffix}` },
  });
  expect(created.ok(), await created.text()).toBeTruthy();

  const consoleErrors: string[] = [];
  const graphPage = await context.newPage();
  for (const target of [page, graphPage]) {
    target.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    target.on('pageerror', (error) => consoleErrors.push(error.message));
  }

  await graphPage.goto('/');
  await expect(graphPage.getByRole('heading', { name: 'Graph Editor' })).toBeVisible();
  await graphPage.locator('#project-select').selectOption(projectId);
  await expect(graphPage.locator('.flow-node')).toHaveCount(0);

  await page.goto('/');
  await page.getByRole('button', { name: 'Agent' }).click();
  await expect(page.getByRole('heading', { name: 'Agent' })).toBeVisible();
  await page.getByLabel('Agent project').selectOption(projectId);
  await page.getByRole('button', { name: 'New agent session' }).click();
  await expect(page.getByText('Graph agent', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const approval = page.locator('.agent-approval');
  await expect(approval).toContainText('graph.apply_patch');
  await expect(approval.locator('code')).toHaveText(/^[0-9a-f]{64}$/);
  await approval.getByRole('button', { name: 'Approve agent tool' }).click();

  await expect(graphPage.locator('.flow-node-service')).toHaveCount(1);
  await expect(graphPage.locator('.flow-node-operator')).toHaveCount(1);
  await expect(graphPage.getByText('AI Value Stepper', { exact: true })).toBeVisible();

  await expect(approval).toContainText('project.deploy');
  await expect(approval.locator('code')).toHaveText(/^[0-9a-f]{64}$/);
  await approval.getByRole('button', { name: 'Approve agent tool' }).click();

  await expect(page.locator('.agent-status')).toHaveText('succeeded', { timeout: 20_000 });
  await expect(page.getByText('Deployment result', { exact: true })).toBeVisible();
  await expect(page.getByText('Runtime monitor evidence', { exact: true })).toBeVisible();
  await expect(page.locator('.agent-tool-call').filter({ hasText: 'runtime.observe' })).toContainText('succeeded');
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('agent-workspace.png'), fullPage: true });

  expect(consoleErrors).toEqual([]);
});
