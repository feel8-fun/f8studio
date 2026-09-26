import { expect, test } from '@playwright/test';

test('edits a code state in one popup and invalidates it after node deletion', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile', 'The graph Inspector is hidden on mobile');
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();
  let otherProjectId: string | null = null;
  try {
    await page.getByLabel('Search nodes').fill('f8.pyengine');
    await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.pyengine' }).first().click();
    await page.getByLabel('Search nodes').fill('f8.python_script');
    await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.python_script' }).click();
    const node = page.locator('.react-flow__node.flow-node-operator');
    await expect(node).toHaveCount(1);
    await node.locator('.node-drag-handle').click();
    const inspector = page.getByRole('complementary', { name: 'Inspector' });
    await expect(inspector.getByRole('button', { name: 'Open code editor for Code' })).toBeVisible();
    await expect(inspector.locator('.state-code-launcher textarea')).toHaveCount(0);

    const opened = page.waitForEvent('popup');
    await inspector.getByRole('button', { name: 'Open code editor for Code' }).click();
    const popup = await opened;
    await expect(popup.getByRole('region', { name: 'Code state editor' })).toBeVisible();
    await expect(popup.getByRole('button', { name: 'Analyze' })).toHaveCount(0);
    await expect(popup.getByRole('complementary', { name: 'Diagnostics' })).toHaveCount(0);
    await expect(popup.getByRole('status').last()).toHaveText('Ready');
    await popup.screenshot({ path: testInfo.outputPath('code-state-editor.png') });
    await popup.locator('.monaco-editor .view-lines').click();
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('print("from popup")');
    await expect.poll(async () => popup.locator('.monaco-editor .view-lines').evaluate((element) => {
      const tokens = [...element.querySelectorAll('span')].filter((item) => item.textContent?.trim());
      return new Set(tokens.map((item) => getComputedStyle(item).color)).size;
    })).toBeGreaterThan(1);
    await expect(popup.getByRole('button', { name: 'Save' })).toBeEnabled();
    await inspector.getByRole('button', { name: 'Open code editor for Code' }).click();
    await expect(popup.locator('.monaco-editor .view-lines')).toContainText('print("from popup")');
    expect(page.context().pages().filter((item) => item.url().includes('view=code-state'))).toHaveLength(1);
    const duplicate = await page.context().newPage();
    await duplicate.goto(popup.url());
    await expect(duplicate.getByRole('status').last()).toHaveText('This code field is already open in another window');
    await expect(duplicate.getByRole('button', { name: 'Save' })).toBeDisabled();
    await duplicate.close();
    await popup.locator('.monaco-editor .view-lines').click();
    await popup.keyboard.press('ControlOrMeta+S');
    await expect(popup.getByRole('status').last()).toHaveText('Saved');
    await popup.locator('.monaco-editor .view-lines').click();
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('value: int = "wrong"');
    await expect(popup.locator('.monaco-editor .squiggly-error').first()).toBeVisible({ timeout: 20000 });
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('print("from popup")');
    await expect(popup.locator('.monaco-editor .squiggly-error')).toHaveCount(0, { timeout: 20000 });
    await expect(popup.getByRole('button', { name: 'Save' })).toBeDisabled();
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('from f8_script_api import F8PyEngineContext\ndef onStart(ctx: F8PyEngineContext) -> None:\n    ctx.');
    await expect(popup.locator('.suggest-widget.visible')).toBeVisible({ timeout: 15000 });
    await expect(popup.locator('.suggest-widget.visible')).toContainText('log');
    await popup.locator('.suggest-widget.visible .monaco-list-row').filter({ hasText: 'log' }).first().click();
    await popup.keyboard.type('(');
    await expect(popup.locator('.parameter-hints-widget.visible')).toBeVisible({ timeout: 15000 });
    await expect(popup.locator('.parameter-hints-widget.visible')).toContainText('message');
    await popup.keyboard.type('"ready")');
    await expect(popup.locator('.monaco-editor .squiggly-error')).toHaveCount(0, { timeout: 20000 });
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('from f8_script_api import F8PyEngineContext\ndef onStart(ctx: F8PyEngineContext) -> None:\n    ctx.emit(');
    await expect(popup.locator('.parameter-hints-widget.visible')).toContainText('port', { timeout: 15000 });
    await popup.keyboard.type('"out", ');
    await expect(popup.locator('.parameter-hints-widget.visible')).toContainText('value', { timeout: 15000 });
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('from f8_script_api import F8PyEngineContext\ndef onStart(ctx: F8PyEngineContext) -> None:\n    ctx.emit()');
    await expect(popup.locator('.parameter-hints-widget.visible')).toHaveCount(0);
    await popup.keyboard.press('ArrowLeft');
    await expect(popup.locator('.parameter-hints-widget.visible')).toContainText('port', { timeout: 15000 });
    await popup.keyboard.press('ControlOrMeta+A');
    await popup.keyboard.type('print("from popup")');
    const nodeId = await node.getAttribute('data-id');
    if (nodeId === null) throw new Error('Missing Python Script node ID');
    await expect.poll(async () => {
      const response = await page.request.get(`/api/projects/${projectId}`);
      const record = await response.json() as { document: { nodes: { nodeId: string; stateValues: Record<string, unknown> }[] } };
      return record.document.nodes.find((item) => item.nodeId === nodeId)?.stateValues.code;
    }).toBe('print("from popup")');

    await expect(popup).toHaveURL(new RegExp(`project=${projectId}`));
    await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
    await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(projectId);
    otherProjectId = await page.locator('#project-select').inputValue();
    await expect(popup).toHaveURL(new RegExp(`project=${projectId}`));
    await expect(popup.locator('.code-state-target')).toContainText('Python Script');
    await page.locator('#project-select').selectOption(projectId);
    await node.locator('.node-drag-handle').click();
    await expect(inspector.getByRole('button', { name: 'Delete node' })).toBeVisible();

    await inspector.getByRole('button', { name: 'Delete node' }).click();
    await expect(popup.getByRole('status').last()).toContainText('no longer exists', { timeout: 10000 });
    await expect(popup.getByRole('button', { name: 'Save' })).toBeDisabled();
    await popup.close();
  } finally {
    if (otherProjectId !== null) await page.request.delete(`/api/projects/${otherProjectId}`);
    await page.request.delete(`/api/projects/${projectId}`);
  }
});
