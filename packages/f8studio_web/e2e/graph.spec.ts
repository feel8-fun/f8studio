import { expect, test, type Page } from '@playwright/test';

async function connectHandles(
  page: Page,
  sourceNodeId: string,
  sourcePortId: string,
  targetNodeId: string,
  targetPortId: string,
): Promise<void> {
  const source = page.locator(`[data-nodeid="${sourceNodeId}"][data-handleid="${sourcePortId}"]`);
  const target = page.locator(`[data-nodeid="${targetNodeId}"][data-handleid="${targetPortId}"]`);
  await source.click({ force: true });
  await target.click({ force: true });
}

async function viewportTransform(page: Page): Promise<string> {
  return page.locator('.react-flow__viewport').evaluate(
    (element) => (element as HTMLElement).style.transform,
  );
}

async function viewportZoom(page: Page): Promise<number> {
  return page.locator('.react-flow__viewport').evaluate((element) => {
    const transform = (element as HTMLElement).style.transform;
    const match = /scale\(([^)]+)\)/.exec(transform);
    if (match?.[1] === undefined) throw new Error(`Viewport transform has no scale: ${transform}`);
    return Number(match[1]);
  });
}

async function zoomOutViewport(page: Page): Promise<string> {
  const initial = await viewportTransform(page);
  await page.locator('.react-flow__controls-zoomout').click();
  await expect.poll(() => viewportTransform(page)).not.toBe(initial);
  return viewportTransform(page);
}

async function observeNodeStability(page: Page, nodeId: string): Promise<void> {
  await page.evaluate((observedNodeId) => {
    const element = document.querySelector<HTMLElement>(`.react-flow__node[data-id="${observedNodeId}"]`);
    if (element === null) throw new Error(`Cannot observe missing React Flow node ${observedNodeId}`);
    const state = { element, hiddenTransitions: 0, observer: null as MutationObserver | null };
    state.observer = new MutationObserver(() => {
      if (element.style.visibility === 'hidden') state.hiddenTransitions += 1;
    });
    state.observer.observe(element, { attributes: true, attributeFilter: ['style'] });
    const testWindow = window as typeof window & {
      nodeStability?: typeof state;
    };
    testWindow.nodeStability?.observer?.disconnect();
    testWindow.nodeStability = state;
  }, nodeId);
}

async function expectObservedNodeStable(page: Page): Promise<void> {
  const result = await page.evaluate(() => {
    const testWindow = window as typeof window & {
      nodeStability?: {
        readonly element: HTMLElement;
        readonly hiddenTransitions: number;
        readonly observer: MutationObserver | null;
      };
    };
    const state = testWindow.nodeStability;
    if (state === undefined) throw new Error('Node stability observer was not installed');
    state.observer?.disconnect();
    delete testWindow.nodeStability;
    return { connected: state.element.isConnected, hiddenTransitions: state.hiddenTransitions };
  });
  expect(result).toEqual({ connected: true, hiddenTransitions: 0 });
}

async function sidePanelVisualState(page: Page): Promise<unknown> {
  return page.evaluate(() => {
    const controls = (selector: string) => [...document.querySelectorAll<HTMLElement>(selector)].map((element) => ({
      disabled: element.matches(':disabled'),
      opacity: getComputedStyle(element).opacity,
    }));
    return {
      rail: controls('.rail button'),
      palette: controls('.graph-palette button, .graph-palette input, .graph-palette select'),
      inspector: controls('.graph-inspector button, .graph-inspector input, .graph-inspector select, .graph-inspector textarea'),
    };
  });
}

test('creates a graph node and restores the persisted project after reload', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();

  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect(page.locator('.save-state')).toHaveText('Saved');
  const projectId = await page.locator('#project-select').inputValue();
  expect(projectId).not.toBe('');

  await page.getByLabel('Search nodes').fill('f8.pyengine');
  await page.getByRole('button', { name: /PyEngine/ }).click();
  await expect(page.locator('.studio-node')).toContainText('PyEngine');
  await expect(page.locator('.graph-toolbar')).toContainText('Draft r1');
  await expect(page.locator('.graph-toolbar')).toContainText('Layout r1');

  await page.reload();
  await expect(page.locator('.connection-online')).toBeVisible();
  await expect(page.locator('#project-select')).toHaveValue(projectId);
  await expect(page.locator('.studio-node')).toContainText('PyEngine');

  await page.evaluate(async (selectedProjectId) => {
    const projectResponse = await fetch(`/api/projects/${selectedProjectId}`);
    const project = await projectResponse.json() as {
      document: { graphRevision: number; layoutRevision: number; nodes: { nodeId: string }[] };
    };
    const node = project.document.nodes[0];
    if (node === undefined) throw new Error('Expected a graph node before external update test');
    const response = await fetch(`/api/projects/${selectedProjectId}/patch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        requestId: crypto.randomUUID(),
        expectedGraphRevision: project.document.graphRevision,
        expectedLayoutRevision: project.document.layoutRevision,
        operations: [{ op: 'renameNode', nodeId: node.nodeId, name: 'PyEngine External' }],
      }),
    });
    if (!response.ok) throw new Error(`External patch failed with HTTP ${response.status}`);
  }, projectId);

  await expect(page.locator('.studio-node')).toContainText('PyEngine External');
  await page.locator('.studio-node').click();
  await page.getByRole('button', { name: 'Duplicate selection' }).click();
  await expect(page.locator('.studio-node')).toHaveCount(2);
  await expect(page.locator('.graph-toolbar')).toContainText('Draft r3');
  await page.reload();
  await expect(page.locator('.studio-node')).toHaveCount(2);
  expect(pageErrors).toEqual([]);
});

test('resizes service canvases and restores their persisted dimensions', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile', 'Precise mouse resizing is covered on desktop');
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();

  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();
  await page.getByLabel('Search nodes').fill('f8.pyengine');
  await page.getByRole('button', { name: /PyEngine/ }).click();
  await page.getByLabel('Search nodes').fill('Bandpass Filter');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'Bandpass Filter' }).click();

  const service = page.locator('.react-flow__node.flow-node-service');
  const operator = page.locator('.react-flow__node.flow-node-operator');
  const resizeHandles = service.locator('.service-resize-handle');
  await expect(service).toHaveCount(1);
  await expect(operator).toHaveCount(1);
  await expect(resizeHandles).toHaveCount(0);
  await service.locator('.node-drag-handle').click({ position: { x: 24, y: 16 } });
  await expect(resizeHandles).toHaveCount(4);

  const before = await page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: {
        nodes: { nodeId: string; kind: string }[];
        layout: { nodeId: string; width?: number | null; height?: number | null }[];
      };
    };
    const serviceNode = record.document.nodes.find((node) => node.kind === 'service');
    const layout = record.document.layout.find((item) => item.nodeId === serviceNode?.nodeId);
    if (serviceNode === undefined || layout?.width == null || layout.height == null) {
      throw new Error('Expected persisted service dimensions before resize');
    }
    return { nodeId: serviceNode.nodeId, width: layout.width, height: layout.height };
  }, projectId);
  const viewportBefore = await viewportTransform(page);
  const bottomRight = service.locator('.service-resize-handle.bottom.right');
  const handleBox = await bottomRight.boundingBox();
  if (handleBox === null) throw new Error('Expected visible bottom-right service resize handle');
  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x + handleBox.width / 2 + 150, handleBox.y + handleBox.height / 2 + 90, { steps: 12 });
  await page.mouse.up();

  await expect.poll(async () => page.evaluate(async ({ selectedProjectId, nodeId, width, height }) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: { layout: { nodeId: string; width?: number | null; height?: number | null }[] };
    };
    const layout = record.document.layout.find((item) => item.nodeId === nodeId);
    return layout !== undefined && (layout.width ?? 0) > width + 50 && (layout.height ?? 0) > height + 30;
  }, { selectedProjectId: projectId, ...before })).toBe(true);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  await expect.poll(() => viewportTransform(page)).toBe(viewportBefore);

  const persisted = await page.evaluate(async ({ selectedProjectId, nodeId }) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: { layout: { nodeId: string; width?: number | null; height?: number | null }[] };
    };
    const layout = record.document.layout.find((item) => item.nodeId === nodeId);
    if (layout?.width == null || layout.height == null) throw new Error('Missing resized service layout');
    return { width: layout.width, height: layout.height };
  }, { selectedProjectId: projectId, nodeId: before.nodeId });
  await expect.poll(() => service.evaluate((element) => ({
    width: Number.parseFloat((element as HTMLElement).style.width),
    height: Number.parseFloat((element as HTMLElement).style.height),
  }))).toEqual(persisted);
  await expect.poll(async () => {
    const [serviceBox, operatorBox] = await Promise.all([service.boundingBox(), operator.boundingBox()]);
    if (serviceBox === null || operatorBox === null) return false;
    return operatorBox.x >= serviceBox.x && operatorBox.y >= serviceBox.y &&
      operatorBox.x + operatorBox.width <= serviceBox.x + serviceBox.width &&
      operatorBox.y + operatorBox.height <= serviceBox.y + serviceBox.height;
  }).toBe(true);

  await page.reload();
  await expect(page.locator('.connection-online')).toBeVisible();
  await expect.poll(() => service.evaluate((element) => ({
    width: Number.parseFloat((element as HTMLElement).style.width),
    height: Number.parseFloat((element as HTMLElement).style.height),
  }))).toEqual(persisted);
  await page.screenshot({ path: testInfo.outputPath('resized-service-canvas.png'), fullPage: true });
  expect(pageErrors).toEqual([]);
});

test('nests operators in compatible services and cascades container deletion', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();

  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();
  await page.getByLabel('Search nodes').fill('f8.pyengine');
  const pyEngineButton = page.getByRole('button', { name: /PyEngine/ });
  await pyEngineButton.click();
  await pyEngineButton.click();
  await expect.poll(async () => page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as { document: { nodes: { kind: string }[] } };
    return record.document.nodes.filter((node) => node.kind === 'service').length;
  }, projectId)).toBe(2);

  await page.getByLabel('Search nodes').fill('Bandpass Filter');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'Bandpass Filter' }).click();
  const services = page.locator('.react-flow__node.flow-node-service');
  const operator = page.locator('.react-flow__node.flow-node-operator');
  await expect(operator).toHaveCount(1);
  await expect(services.nth(0).locator('.service-child-count')).toHaveText('1 ops');

  const graph = await page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    if (!response.ok) throw new Error(`Project fetch failed with HTTP ${response.status}`);
    const record = await response.json() as {
      document: {
        graphRevision: number;
        layoutRevision: number;
        nodes: { kind: 'service' | 'operator'; nodeId: string; serviceId: string }[];
        layout: { nodeId: string; x: number; y: number }[];
      };
    };
    const serviceNodes = record.document.nodes.filter((node) => node.kind === 'service');
    const operatorNode = record.document.nodes.find((node) => node.kind === 'operator');
    if (serviceNodes.length !== 2 || operatorNode === undefined) throw new Error('Expected two services and one operator');
    return { document: record.document, serviceNodes, operatorNode };
  }, projectId);

  const secondServiceId = graph.serviceNodes[1]!.serviceId;
  if ((page.viewportSize()?.width ?? 0) > 560) {
    await zoomOutViewport(page);
    const zoomedScale = await viewportZoom(page);
    await observeNodeStability(page, graph.operatorNode.nodeId);
    const [operatorBox, targetBox] = await Promise.all([
      operator.locator('.node-drag-handle').boundingBox(),
      services.nth(1).boundingBox(),
    ]);
    if (operatorBox === null || targetBox === null) throw new Error('Expected visible graph nodes before drag');
    await page.mouse.move(operatorBox.x + operatorBox.width / 2, operatorBox.y + operatorBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(targetBox.x + targetBox.width / 2, targetBox.y + targetBox.height / 2, { steps: 12 });
    await page.mouse.up();
    await expect(services.nth(1).locator('.service-child-count')).toHaveText('1 ops');
    await expect(page.locator('.save-state')).toHaveText('Saved');
    await expect.poll(() => viewportZoom(page)).toBe(zoomedScale);
    await expectObservedNodeStable(page);
    await operator.locator('.studio-node').click();
    await expect(page.getByLabel('Service binding')).toHaveValue(secondServiceId);
    const beforeMove = await page.evaluate(async ({ selectedProjectId, serviceId, operatorId }) => {
      const response = await fetch(`/api/projects/${selectedProjectId}`);
      const record = await response.json() as {
        document: { layout: { nodeId: string; x: number; y: number }[] };
      };
      const serviceLayout = record.document.layout.find((layout) => layout.nodeId === serviceId);
      const operatorLayout = record.document.layout.find((layout) => layout.nodeId === operatorId);
      if (serviceLayout === undefined || operatorLayout === undefined) throw new Error('Missing layout before service drag');
      return { serviceLayout, operatorLayout };
    }, { selectedProjectId: projectId, serviceId: secondServiceId, operatorId: graph.operatorNode.nodeId });
    const serviceHeaderBox = await services.nth(1).locator('.node-drag-handle').boundingBox();
    if (serviceHeaderBox === null) throw new Error('Expected visible service header before drag');
    const dragStart = {
      x: serviceHeaderBox.x + 24,
      y: serviceHeaderBox.y + serviceHeaderBox.height / 2,
    };
    await observeNodeStability(page, graph.serviceNodes[1]!.nodeId);
    await page.mouse.move(dragStart.x, dragStart.y);
    await page.mouse.down();
    await page.mouse.move(dragStart.x + 60, dragStart.y + 40, { steps: 12 });
    await page.mouse.up();
    await expect.poll(async () => page.evaluate(async ({ selectedProjectId, serviceId, operatorId, before }) => {
      const response = await fetch(`/api/projects/${selectedProjectId}`);
      const record = await response.json() as {
        document: { layout: { nodeId: string; x: number; y: number }[] };
      };
      const serviceLayout = record.document.layout.find((layout) => layout.nodeId === serviceId);
      const operatorLayout = record.document.layout.find((layout) => layout.nodeId === operatorId);
      if (serviceLayout === undefined || operatorLayout === undefined) return false;
      const serviceDelta = {
        x: serviceLayout.x - before.serviceLayout.x,
        y: serviceLayout.y - before.serviceLayout.y,
      };
      const operatorDelta = {
        x: operatorLayout.x - before.operatorLayout.x,
        y: operatorLayout.y - before.operatorLayout.y,
      };
      return Math.abs(serviceDelta.x) > 1 && Math.abs(serviceDelta.y) > 1 &&
        Math.abs(serviceDelta.x - operatorDelta.x) < 0.01 &&
        Math.abs(serviceDelta.y - operatorDelta.y) < 0.01;
    }, {
      selectedProjectId: projectId,
      serviceId: secondServiceId,
      operatorId: graph.operatorNode.nodeId,
      before: beforeMove,
    })).toBe(true);
    await expect.poll(() => viewportZoom(page)).toBe(zoomedScale);
    await expectObservedNodeStable(page);
    await page.locator('.react-flow__controls-fitview').click();
    await expect(operator).toBeVisible();
  } else {
    await page.evaluate(async ({ selectedProjectId, document, operatorNode, targetServiceId }) => {
      const targetLayout = document.layout.find((layout) => layout.nodeId === targetServiceId);
      if (targetLayout === undefined) throw new Error('Missing target service layout');
      const response = await fetch(`/api/projects/${selectedProjectId}/patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requestId: crypto.randomUUID(),
          expectedGraphRevision: document.graphRevision,
          expectedLayoutRevision: document.layoutRevision,
          operations: [
            { op: 'bindOperatorService', nodeId: operatorNode.nodeId, serviceId: targetServiceId },
            {
              op: 'setNodeLayout',
              layout: { nodeId: operatorNode.nodeId, x: targetLayout.x + 32, y: targetLayout.y + 96, collapsed: false },
            },
          ],
        }),
      });
      if (!response.ok) throw new Error(`Operator rebind failed with HTTP ${response.status}`);
    }, {
      selectedProjectId: projectId,
      document: graph.document,
      operatorNode: graph.operatorNode,
      targetServiceId: secondServiceId,
    });
    await page.reload();
    await expect(page.locator('.connection-online')).toBeVisible();
  }

  await expect(services.nth(0).locator('.service-child-count')).toHaveCount(0);
  await expect(services.nth(1).locator('.service-child-count')).toHaveText('1 ops');
  await expect.poll(async () => {
    const [serviceBox, operatorBox] = await Promise.all([services.nth(1).boundingBox(), operator.boundingBox()]);
    if (serviceBox === null || operatorBox === null) return false;
    return operatorBox.x >= serviceBox.x && operatorBox.y >= serviceBox.y &&
      operatorBox.x + operatorBox.width <= serviceBox.x + serviceBox.width &&
      operatorBox.y + operatorBox.height <= serviceBox.y + serviceBox.height;
  }).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('service-nesting.png'), fullPage: true });

  await services.nth(1).locator('.node-drag-handle').click({ force: true, position: { x: 2, y: 2 } });
  await page.keyboard.press('Delete');
  await expect(services).toHaveCount(1);
  await expect(operator).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test('edits inline state and configures typed exec and data connections', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();

  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();

  await page.getByLabel('Search nodes').fill('f8.pyengine');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.pyengine' }).first().click();
  await page.getByLabel('Search nodes').fill('f8.detrend');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.detrend' }).click();
  const inlineSlider = page.locator('.flow-node-operator .state-control-inline input[type="range"]');
  await expect(inlineSlider).toHaveCount(1);
  await inlineSlider.fill('0.75');
  await inlineSlider.dispatchEvent('pointerup');
  await expect.poll(async () => page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as { document: { nodes: { operatorClass?: string; stateValues: Record<string, unknown> }[] } };
    return record.document.nodes.find((node) => node.operatorClass === 'f8.detrend')?.stateValues.alpha;
  }, projectId)).toBe(0.75);

  await page.getByLabel('Search nodes').fill('f8.tick');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.tick' }).click();
  await page.getByLabel('Search nodes').fill('f8.print');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.print' }).click();
  await expect(page.locator('.react-flow__node.flow-node-operator')).toHaveCount(3);

  const graph = await page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: {
        nodes: {
          nodeId: string;
          operatorClass?: string;
          ports: { portId: string; name: string; kind: string; direction: string }[];
        }[];
      };
    };
    const detrend = record.document.nodes.find((node) => node.operatorClass === 'f8.detrend');
    const tick = record.document.nodes.find((node) => node.operatorClass === 'f8.tick');
    const print = record.document.nodes.find((node) => node.operatorClass === 'f8.print');
    if (detrend === undefined || tick === undefined || print === undefined) throw new Error('Expected Detrend, Tick, and Print operators');
    const findPort = (node: typeof tick, name: string, kind: string, direction: string) => {
      const found = node.ports.find((port) => port.name === name && port.kind === kind && port.direction === direction);
      if (found === undefined) throw new Error(`Missing ${node.operatorClass}.${name} port`);
      return found.portId;
    };
    return {
      tickId: tick.nodeId,
      detrendId: detrend.nodeId,
      printId: print.nodeId,
      tickExec: findPort(tick, 'exec', 'exec', 'output'),
      printExec: findPort(print, 'exec', 'exec', 'input'),
      detrendData: findPort(detrend, 'value', 'data', 'output'),
      printData: findPort(print, 'value', 'data', 'input'),
    };
  }, projectId);

  const zoomedViewport = await zoomOutViewport(page);
  await observeNodeStability(page, graph.tickId);
  const sidePanelsBeforeSave = await sidePanelVisualState(page);
  let releasePatch: (() => void) | null = null;
  await page.route('**/api/projects/*/patch', async (route) => {
    await new Promise<void>((resolve) => {
      releasePatch = resolve;
    });
    await route.continue();
  }, { times: 1 });
  await connectHandles(page, graph.tickId, graph.tickExec, graph.printId, graph.printExec);
  await expect(page.locator('.save-state')).toHaveText('Saving...');
  await expect.poll(() => releasePatch !== null).toBe(true);
  try {
    expect(await sidePanelVisualState(page)).toEqual(sidePanelsBeforeSave);
  } finally {
    releasePatch?.();
  }
  await expect(page.locator('.react-flow__edge.graph-edge-exec')).toHaveCount(1);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  await expect.poll(() => viewportTransform(page)).toBe(zoomedViewport);
  await connectHandles(page, graph.detrendId, graph.detrendData, graph.printId, graph.printData);
  const dataEdge = page.locator('.react-flow__edge.graph-edge-data');
  await expect(dataEdge).toHaveCount(1);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  await expect.poll(() => viewportTransform(page)).toBe(zoomedViewport);
  await expectObservedNodeStable(page);

  if ((page.viewportSize()?.width ?? 0) > 560) {
    await dataEdge.click({ force: true });
    await expect(page.locator('.graph-inspector')).toContainText('data');
    await page.getByLabel('Delivery').selectOption('queue');
    await expect(page.locator('.save-state')).toHaveText('Saved');
    const queueSize = page.getByLabel('Queue size');
    await queueSize.fill('8');
    await queueSize.press('Tab');
    const timeout = page.getByLabel('Stale timeout (ms)');
    await timeout.fill('500');
    await timeout.press('Tab');
    await expect.poll(async () => page.evaluate(async (selectedProjectId) => {
      const response = await fetch(`/api/projects/${selectedProjectId}`);
      const record = await response.json() as {
        document: { edges: { kind: string; strategy: string; queueSize: number; timeoutMs: number | null }[] };
      };
      const edge = record.document.edges.find((candidate) => candidate.kind === 'data');
      return edge === undefined ? null : `${edge.strategy}:${edge.queueSize}:${edge.timeoutMs}`;
    }, projectId)).toBe('queue:8:500');
  }
  await page.screenshot({ path: testInfo.outputPath('typed-connections.png'), fullPage: true });
  expect(pageErrors).toEqual([]);
});

test('keeps long inline state editors within fixed node rows', async ({ page }, testInfo) => {
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);

  await page.getByLabel('Search nodes').fill('f8.pyengine');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.pyengine' }).first().click();
  await page.getByLabel('Search nodes').fill('f8.state_expr');
  await page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'f8.state_expr' }).first().click();

  const operator = page.locator('.react-flow__node.flow-node-operator');
  await expect(operator).toHaveCount(1);
  await expect(operator.locator('.state-control-inline textarea')).toHaveCount(0);
  const codeInput = operator.locator('.state-text-code input');
  await expect(codeInput).toHaveCount(1);
  await codeInput.fill('input_value + another_input_value + a_very_long_expression_name_that_must_not_resize_the_node');
  await codeInput.press('Tab');
  await expect(page.locator('.save-state')).toHaveText('Saved');
  await expect.poll(async () => {
    const box = await operator.boundingBox();
    return box?.height ?? null;
  }).toBeLessThanOrEqual(140);
  await expect.poll(async () => operator.evaluate((element) => ({
    width: Number.parseFloat(getComputedStyle(element).width),
    overflow: element.scrollWidth - element.clientWidth,
  }))).toEqual({ width: 240, overflow: 0 });
  await page.screenshot({ path: testInfo.outputPath('compact-inline-state.png'), fullPage: true });
});

test('keeps one RW state label and preserves its editor when connected', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();

  await page.getByLabel('Search nodes').fill('f8.pyengine');
  await page.getByRole('button', { name: /PyEngine/ }).click();
  await page.getByLabel('Search nodes').fill('Bandpass Filter');
  const bandpassButton = page.locator('.catalog-list button:not(:disabled)').filter({ hasText: 'Bandpass Filter' });
  await bandpassButton.click();
  await expect(page.locator('.react-flow__node.flow-node-operator')).toHaveCount(1);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  await bandpassButton.click();
  await expect(page.locator('.react-flow__node.flow-node-operator')).toHaveCount(2);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  const statePorts = await page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: {
        nodes: {
          nodeId: string;
          operatorClass?: string;
          ports: { portId: string; runtimeName: string; kind: string; direction: string }[];
        }[];
      };
    };
    const operators = record.document.nodes.filter((node) => node.operatorClass === 'f8.bandpass_filter');
    if (operators.length !== 2) throw new Error('Expected two Bandpass Filter operators');
    const port = (node: typeof operators[number], direction: string) => {
      const found = node.ports.find((candidate) =>
        candidate.kind === 'state' && candidate.runtimeName === 'low_cutoff' && candidate.direction === direction);
      if (found === undefined) throw new Error(`Missing low_cutoff ${direction} port`);
      return found.portId;
    };
    return {
      sourceNodeId: operators[0]!.nodeId,
      sourcePortId: port(operators[0]!, 'output'),
      targetNodeId: operators[1]!.nodeId,
      targetPortId: port(operators[1]!, 'input'),
    };
  }, projectId);
  const targetNode = page.locator(`.react-flow__node[data-id="${statePorts.targetNodeId}"]`);
  const targetRow = targetNode.locator('.port-row-shared-state').filter({ hasText: 'low_cutoff' });
  await expect(targetRow).toHaveCount(1);
  await expect(targetRow.getByText('low_cutoff', { exact: true })).toHaveCount(1);
  const editor = targetRow.locator('input[type="number"]');
  await expect(editor).toHaveCount(1);
  const editorElement = await editor.elementHandle();
  if (editorElement === null) throw new Error('Expected numeric state editor before connection');

  await connectHandles(
    page,
    statePorts.sourceNodeId,
    statePorts.sourcePortId,
    statePorts.targetNodeId,
    statePorts.targetPortId,
  );
  await expect(page.locator('.react-flow__edge.graph-edge-state')).toHaveCount(1);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  await expect(editor).toHaveAttribute('readonly', '');
  expect(await editor.evaluate((element, before) => element === before, editorElement)).toBe(true);
  await expect(targetRow.getByText('low_cutoff', { exact: true })).toHaveCount(1);
  await page.screenshot({ path: testInfo.outputPath('connected-state-editor.png'), fullPage: true });
  expect(pageErrors).toEqual([]);
});

test('builds, deploys, observes, modifies, and restores a built-in Studio graph', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile', 'Runtime observation uses the desktop Inspector');
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();

  const previousProjectId = await page.locator('#project-select').inputValue();
  await page.locator('.project-control').getByRole('button', { name: 'New project' }).click();
  await expect.poll(() => page.locator('#project-select').inputValue()).not.toBe(previousProjectId);
  const projectId = await page.locator('#project-select').inputValue();

  await page.getByLabel('Search nodes').fill('f8.value_stepper');
  const stepperButton = page.locator('.catalog-list button').filter({ hasText: 'f8.value_stepper' });
  await expect(stepperButton).toBeEnabled();
  await stepperButton.click();
  await expect(page.locator('.react-flow__node.flow-node-service')).toHaveCount(1);
  await expect(page.locator('.react-flow__node.flow-node-operator')).toHaveCount(1);
  await expect(page.locator('.graph-toolbar')).toContainText('Draft r1 · Layout r1');
  await expect.poll(async () => page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: { nodes: { kind: string; nodeId: string; serviceId: string; operatorClass?: string }[] };
    };
    const service = record.document.nodes.find((node) => node.kind === 'service');
    const operator = record.document.nodes.find((node) => node.operatorClass === 'f8.value_stepper');
    return service?.nodeId === 'studio' && operator?.serviceId === 'studio';
  }, projectId)).toBe(true);

  const slider = page.locator('.flow-node-operator .state-control-inline input[type="range"]');
  await slider.fill('0.6');
  await slider.dispatchEvent('pointerup');
  await expect(page.locator('.graph-toolbar')).toContainText('Draft r2');
  await expect.poll(async () => page.evaluate(async (selectedProjectId) => {
    const response = await fetch(`/api/projects/${selectedProjectId}`);
    const record = await response.json() as {
      document: { nodes: { operatorClass?: string; stateValues: Record<string, unknown> }[] };
    };
    return record.document.nodes.find((node) => node.operatorClass === 'f8.value_stepper')?.stateValues.value;
  }, projectId)).toBe(0.6);

  await page.getByRole('button', { name: 'Deploy' }).click();
  await expect(page.locator('.deploy-state')).toContainText('succeeded r2', { timeout: 20_000 });
  await page.locator('.flow-node-service .node-drag-handle').click({ force: true, position: { x: 24, y: 12 } });
  await expect(page.locator('.graph-inspector .monitor-values')).toContainText('Ready', { timeout: 10_000 });

  await slider.fill('0.75');
  await slider.dispatchEvent('pointerup');
  await expect(page.locator('.graph-toolbar')).toContainText('Draft r3');
  await expect(page.locator('.deploy-state')).toContainText('succeeded r2');
  await page.reload();
  await expect(page.locator('#project-select')).toHaveValue(projectId);
  await expect(page.locator('.flow-node-operator .state-control-inline input[type="range"]')).toHaveValue('0.75');
  await expect(page.locator('.deploy-state')).toContainText('succeeded r2');
  await page.screenshot({ path: testInfo.outputPath('studio-runtime-workflow.png'), fullPage: true });
  expect(pageErrors).toEqual([]);
});
