import { expect, test } from '@playwright/test';

test('agent builds a graph with exact approvals and updates an open graph viewer', async ({ page, request }, testInfo) => {
  const suffix = `${testInfo.project.name}-${Date.now()}`.replaceAll(/[^a-zA-Z0-9_-]/g, '-');
  const projectId = `agent-e2e-${suffix}`;
  const created = await request.post('/api/projects', {
    data: { projectId, name: `Agent E2E ${suffix}` },
  });
  expect(created.ok(), await created.text()).toBeTruthy();

  try {
    const consoleErrors: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => consoleErrors.push(error.message));

    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Graph Editor' })).toBeVisible();
    await page.locator('#project-select').selectOption(projectId);
    await expect(page.locator('.flow-node')).toHaveCount(0);

    const opened = page.waitForEvent('popup');
    await page.getByRole('button', { name: 'Open Agent window' }).click();
    const agentPage = await opened;
    if (testInfo.project.name === 'mobile') await agentPage.setViewportSize({ width: 390, height: 844 });
    agentPage.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    agentPage.on('pageerror', (error) => consoleErrors.push(error.message));
    await expect(agentPage.getByRole('heading', { name: 'Agent' })).toBeVisible();
    if (testInfo.project.name === 'mobile') await expect.poll(() => agentPage.locator('.agent-conversation').evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.left >= 0 && bounds.right <= innerWidth + 1 && bounds.width >= 300;
    })).toBe(true);
    await agentPage.getByLabel('Agent provider', { exact: true }).selectOption('deterministic');
    await agentPage.getByRole('button', { name: 'New agent session' }).click();
    await expect(agentPage.locator('.agent-current-title')).toHaveText('Studio agent');
    await agentPage.getByRole('textbox', { name: 'Agent prompt' }).fill('Build a graph with a value stepper');

    await agentPage.getByRole('button', { name: 'Run', exact: true }).click();
    const approval = agentPage.locator('.agent-approval');
    await expect(approval).toContainText('graph.apply_patch');
    await expect(approval.locator('code')).toHaveText(/^[0-9a-f]{64}$/);
    await expect(approval.locator('.agent-patch-preview')).toContainText('node');
    await approval.locator('.agent-patch-preview > summary').click();
    await expect(approval.locator('.agent-patch-preview ol li').first()).toBeVisible();
    await expect.poll(() => agentPage.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
    await agentPage.screenshot({ path: testInfo.outputPath('agent-approval.png'), fullPage: true });
    if (testInfo.project.name === 'mobile') await expect.poll(() => approval.getByRole('button', { name: 'Approve agent tool' }).evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.left >= 0 && bounds.right <= innerWidth + 1;
    })).toBe(true);
    await approval.getByRole('button', { name: 'Approve agent tool' }).click();

    await expect(page.locator('.flow-node-service')).toHaveCount(1);
    await expect(page.locator('.flow-node-operator')).toHaveCount(1);
    await expect(page.getByText('AI Value Stepper', { exact: true })).toBeVisible();

    await expect(approval).toContainText('project.deploy');
    await expect(approval.locator('code')).toHaveText(/^[0-9a-f]{64}$/);
    await approval.getByRole('button', { name: 'Approve agent tool' }).click();

    await expect(agentPage.locator('.agent-status')).toHaveText('succeeded', { timeout: 20_000 });
    await expect(agentPage.getByText('Deployment result', { exact: true })).toBeVisible();
    await expect(agentPage.getByText('Runtime monitor evidence', { exact: true })).toBeVisible();
    await expect(agentPage.locator('.agent-tool-call').filter({ hasText: 'runtime.observe' })).toContainText('succeeded');
    await expect.poll(() => agentPage.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
    await agentPage.screenshot({ path: testInfo.outputPath('agent-workspace.png'), fullPage: true });

    await agentPage.getByRole('button', { name: 'Rename session' }).click();
    await agentPage.getByRole('textbox', { name: 'Session title' }).fill('Value stepper graph');
    await agentPage.getByRole('button', { name: 'Save session title' }).click();
    await expect(agentPage.locator('.agent-current-title')).toHaveText('Value stepper graph');
    await agentPage.getByRole('button', { name: 'Delete session' }).click();
    const deleteDialog = agentPage.getByRole('dialog', { name: 'Delete agent session' });
    await expect(deleteDialog).toContainText('The project graph will stay unchanged.');
    await deleteDialog.getByRole('button', { name: 'Delete session' }).click();
    await expect(agentPage.getByText('Create a session for the selected project.')).toBeVisible();
    await expect(page.locator('.flow-node-operator')).toHaveCount(1);

    expect(consoleErrors).toEqual([]);
  } finally {
    const sessions = await request.get(`/api/agents/sessions?project_id=${projectId}`);
    if (sessions.ok()) {
      const records = await sessions.json() as { sessionId: string; status: string }[];
      for (const session of records) {
        if (session.status === 'running' || session.status === 'waiting_for_approval') {
          await request.delete(`/api/agents/sessions/${session.sessionId}/runs/current`);
        }
      }
    }
    await request.post(`/api/projects/${projectId}/stop`);
    const deleted = await request.delete(`/api/projects/${projectId}`);
    expect(deleted.status(), await deleted.text()).toBe(204);
  }
});
