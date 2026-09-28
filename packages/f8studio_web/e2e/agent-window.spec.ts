import { expect, test } from '@playwright/test';

test('agent window stays bound to the graph project and reopens on the same session', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();

  try {
    const opened = page.waitForEvent('popup');
    await page.getByRole('button', { name: 'Open Agent window' }).click();
    const popup = await opened;
    await expect(popup).toHaveURL(new RegExp(`view=agent&project=${projectId}`));
    await expect(popup.getByRole('heading', { name: 'Agent' })).toBeVisible();
    await expect(popup.getByRole('navigation', { name: 'Workspace navigation' })).toHaveCount(0);
    await expect(popup.getByRole('combobox', { name: 'Agent project' })).toHaveCount(0);

    await popup.getByRole('button', { name: 'New agent session' }).click();
    await expect(popup.getByRole('textbox', { name: 'Agent prompt' })).toBeVisible();
    await expect(popup).toHaveURL(/session=/);
    const sessionUrl = popup.url();
    await popup.reload();
    await expect(popup.getByRole('textbox', { name: 'Agent prompt' })).toBeVisible();
    await expect(popup).toHaveURL(sessionUrl);

    await page.getByRole('button', { name: 'Open Agent window' }).click();
    expect(page.context().pages()).toHaveLength(2);
    await expect(popup).toHaveURL(sessionUrl);
    await expect(page.getByRole('heading', { name: 'Graph Editor' })).toBeVisible();
    await popup.close();
  } finally {
    await page.request.delete(`/api/projects/${projectId}`);
  }
});
