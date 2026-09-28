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
    agentPage.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    agentPage.on('pageerror', (error) => consoleErrors.push(error.message));
    await expect(agentPage.getByRole('heading', { name: 'Agent' })).toBeVisible();
    await agentPage.getByLabel('Agent provider').selectOption('deterministic');
    await agentPage.getByRole('button', { name: 'New agent session' }).click();
    await expect(agentPage.getByText('Studio agent', { exact: true })).toBeVisible();
    await agentPage.getByRole('textbox', { name: 'Agent prompt' }).fill('Build a graph with a value stepper');

    await agentPage.getByRole('button', { name: 'Run', exact: true }).click();
    const approval = agentPage.locator('.agent-approval');
    await expect(approval).toContainText('graph.apply_patch');
    await expect(approval.locator('code')).toHaveText(/^[0-9a-f]{64}$/);
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
