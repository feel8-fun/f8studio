import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

import { isGraphNode, type GraphEdge, type GraphNode, type NodeLayout } from '../src/api/contracts';
import { nodePortRows } from '../src/graph/portRows';

const SELECTED_PROJECT_KEY = 'f8studio.selectedProjectId';
const EVIDENCE_PATH = fileURLToPath(new URL('../../../docs/plans/evidence/p4-graph-performance.json', import.meta.url));

interface GraphFixtureResult {
  readonly projectId: string;
  readonly patchMs: number;
  readonly nodeProfile: string;
  readonly visiblePortRows: number;
}

interface GraphScenarioResult {
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly nodeProfile: string;
  readonly visiblePortRows: number;
  readonly patchMs: number;
  readonly renderReadyMs: number;
  readonly renderedNodeCount: number;
  readonly renderedEdgeCount: number;
  readonly usedJsHeapMiB: number | null;
  readonly interactionP95Ms: number;
  readonly dragCommitMs: number;
  readonly dragAverageFps: number;
  readonly dragFrameP95Ms: number;
}

function percentile(values: readonly number[], fraction: number): number {
  if (values.length === 0) throw new Error('Cannot calculate a percentile without samples');
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)]!;
}

function operatorNode(template: GraphNode, index: number): GraphNode {
  if (template.kind !== 'operator') throw new Error('Benchmark operator template is not an operator');
  const nodeId = `benchmark_node_${index}`;
  return {
    ...template,
    nodeId,
    name: `Handy Out ${index}`,
  };
}

function graphEdges(template: GraphNode, operatorCount: number, edgeCount: number): readonly GraphEdge[] {
  const inputs = template.ports.filter((port) => port.kind === 'data' && port.direction === 'input');
  const outputs = template.ports.filter((port) => port.kind === 'data' && port.direction === 'output');
  if (inputs.length === 0 || outputs.length === 0) throw new Error('Benchmark operator requires data input and output ports');
  if (edgeCount > operatorCount * inputs.length) {
    throw new Error(`${operatorCount} benchmark operators cannot accept ${edgeCount} unique input edges`);
  }
  return Array.from({ length: edgeCount }, (_, edgeIndex): GraphEdge => {
    const targetIndex = Math.floor(edgeIndex / inputs.length);
    const portIndex = edgeIndex % inputs.length;
    const sourceIndex = (targetIndex + 1) % operatorCount;
    return {
      edgeId: `benchmark_edge_${edgeIndex}`,
      fromNodeId: `benchmark_node_${sourceIndex}`,
      fromPortId: outputs[portIndex % outputs.length]!.portId,
      toNodeId: `benchmark_node_${targetIndex}`,
      toPortId: inputs[portIndex]!.portId,
      kind: 'data',
      strategy: 'latest',
      queueSize: 16,
      timeoutMs: null,
    };
  });
}

async function createGraphFixture(
  request: APIRequestContext,
  nodeCount: number,
  edgeCount: number,
): Promise<GraphFixtureResult> {
  const createResponse = await request.post('/api/projects', {
    data: { name: `P4 Graph Benchmark ${nodeCount}-${randomUUID().slice(0, 8)}` },
  });
  expect(createResponse.ok()).toBe(true);
  const created: unknown = await createResponse.json();
  if (typeof created !== 'object' || created === null ||
      typeof (created as Record<string, unknown>).projectId !== 'string') {
    throw new Error('Benchmark project response is missing projectId');
  }
  const projectId = (created as { readonly projectId: string }).projectId;
  const [serviceResponse, operatorResponse] = await Promise.all([
    request.post('/api/catalog/nodes', {
      data: { kind: 'service', nodeId: 'benchmark_service', serviceClass: 'f8.pyengine' },
    }),
    request.post('/api/catalog/nodes', {
      data: {
        kind: 'operator',
        nodeId: 'benchmark_template',
        serviceId: 'benchmark_service',
        serviceClass: 'f8.pyengine',
        operatorClass: 'f8.handy_out',
      },
    }),
  ]);
  expect(serviceResponse.ok()).toBe(true);
  expect(operatorResponse.ok()).toBe(true);
  const serviceTemplate: unknown = await serviceResponse.json();
  const operatorTemplate: unknown = await operatorResponse.json();
  if (!isGraphNode(serviceTemplate) || serviceTemplate.kind !== 'service') {
    throw new Error('Benchmark service template does not match f8studio-api/1');
  }
  if (!isGraphNode(operatorTemplate) || operatorTemplate.kind !== 'operator') {
    throw new Error('Benchmark operator template does not match f8studio-api/1');
  }
  const operatorCount = nodeCount - 1;
  const nodes: readonly GraphNode[] = [
    serviceTemplate,
    ...Array.from({ length: operatorCount }, (_, index) => operatorNode(operatorTemplate, index)),
  ];
  const edges = graphEdges(operatorTemplate, operatorCount, edgeCount);
  const layout: readonly NodeLayout[] = [{
    nodeId: 'benchmark_service',
    x: 80,
    y: 80,
    width: 620,
    height: 320,
    collapsed: false,
  }];
  const patchStarted = performance.now();
  const patchResponse = await request.post(`/api/projects/${projectId}/patch`, {
    data: {
      requestId: randomUUID(),
      expectedGraphRevision: 0,
      expectedLayoutRevision: 0,
      operations: [{ op: 'insertFragment', nodes, edges, layout }],
    },
    timeout: 120_000,
  });
  const patchMs = performance.now() - patchStarted;
  if (!patchResponse.ok()) throw new Error(`Benchmark graph patch failed: ${patchResponse.status()} ${await patchResponse.text()}`);
  return {
    projectId,
    patchMs,
    nodeProfile: `${operatorTemplate.serviceClass}/${operatorTemplate.operatorClass}`,
    visiblePortRows: nodePortRows(operatorTemplate).length,
  };
}

async function measureInteractions(page: Page): Promise<readonly number[]> {
  const nodeIds = await interactiveOperatorIds(page);
  if (nodeIds.length < 2) throw new Error(`Expected at least two interactive operators, found ${nodeIds.length}`);
  const samples: number[] = [];
  for (let index = 0; index < 20; index += 1) {
    const nodeId = nodeIds[index % Math.min(nodeIds.length, 10)]!;
    samples.push(await page.evaluate(async (selectedNodeId) => {
      const node = document.querySelector<HTMLElement>(`.react-flow__node[data-id="${selectedNodeId}"]`);
      const expectedName = node?.querySelector('header strong')?.textContent;
      if (node === null || expectedName === null || expectedName === undefined) {
        throw new Error(`Benchmark node is unavailable: ${selectedNodeId}`);
      }
      const bounds = node.getBoundingClientRect();
      const started = performance.now();
      node.dispatchEvent(new MouseEvent('click', {
        bubbles: true,
        clientX: bounds.x + bounds.width / 2,
        clientY: bounds.y + bounds.height / 2,
      }));
      return new Promise<number>((resolve, reject) => {
        const deadline = started + 1000;
        const inspect = (): void => {
          const input = document.querySelector<HTMLInputElement>('.graph-inspector .inspector-field input');
          if (input?.value === expectedName) {
            resolve(performance.now() - started);
            return;
          }
          if (performance.now() >= deadline) {
            reject(new Error(`Inspector did not select ${selectedNodeId}`));
            return;
          }
          requestAnimationFrame(inspect);
        };
        requestAnimationFrame(inspect);
      });
    }, nodeId));
  }
  return samples;
}

async function interactiveOperatorIds(page: Page): Promise<readonly string[]> {
  return page.locator('.react-flow__node.flow-node-operator').evaluateAll((elements) => {
    const flow = document.querySelector('.graph-canvas .react-flow');
    if (flow === null) return [];
    const flowBounds = flow.getBoundingClientRect();
    return elements.flatMap((element) => {
      const bounds = element.getBoundingClientRect();
      const id = element.getAttribute('data-id');
      const interactive = id !== null && bounds.width > 0 && bounds.height > 0 &&
        bounds.top >= flowBounds.top + 4 && bounds.bottom <= flowBounds.bottom - 4 &&
        bounds.left >= flowBounds.left + 4 && bounds.right <= flowBounds.right - 4;
      return interactive ? [id] : [];
    });
  });
}

async function measureDrag(page: Page): Promise<{
  readonly averageFps: number;
  readonly frameP95Ms: number;
  readonly commitMs: number;
}> {
  const nodeIds = await interactiveOperatorIds(page);
  if (nodeIds.length === 0) throw new Error('Expected an interactive operator for drag sampling');
  const node = page.locator(`.react-flow__node[data-id="${nodeIds[Math.floor(nodeIds.length / 2)]}"]`);
  const header = node.locator('header');
  const box = await header.boundingBox();
  if (box === null) throw new Error('Visible benchmark node has no drag bounds');
  await page.evaluate(() => {
    const state = window as Window & { __graphDragFrames?: number[]; __graphDragSampling?: boolean };
    state.__graphDragFrames = [];
    state.__graphDragSampling = true;
    const sample = (timestamp: number): void => {
      if (state.__graphDragSampling !== true) return;
      state.__graphDragFrames?.push(timestamp);
      requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
  });
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;
  await page.mouse.move(centerX, centerY);
  await page.mouse.down();
  for (let step = 0; step < 60; step += 1) {
    await page.mouse.move(centerX + Math.sin(step / 4) * 6, centerY);
    await page.waitForTimeout(16);
  }
  await page.mouse.move(centerX, centerY);
  const patchResponse = page.waitForResponse((response) =>
    response.request().method() === 'POST' && response.url().includes('/patch'), { timeout: 15_000 });
  const commitStarted = performance.now();
  await page.mouse.up();
  await page.evaluate(() => {
    const state = window as Window & { __graphDragSampling?: boolean };
    state.__graphDragSampling = false;
  });
  const response = await patchResponse;
  expect(response.ok()).toBe(true);
  await expect(page.locator('.save-state')).toHaveText('Saved');
  const commitMs = performance.now() - commitStarted;
  const frames = await page.evaluate(() => {
    const state = window as Window & { __graphDragFrames?: number[] };
    return state.__graphDragFrames ?? [];
  });
  const intervals = frames.slice(1).map((timestamp, index) => timestamp - frames[index]!);
  if (intervals.length < 30) throw new Error(`Drag frame sample is too short: ${intervals.length}`);
  const elapsed = frames.at(-1)! - frames[0]!;
  return {
    averageFps: (intervals.length * 1000) / elapsed,
    frameP95Ms: percentile(intervals, 0.95),
    commitMs,
  };
}

async function benchmarkScenario(
  page: Page,
  fixture: GraphFixtureResult,
  nodeCount: number,
  edgeCount: number,
): Promise<GraphScenarioResult> {
  await page.goto('/api/health');
  await page.evaluate(([key, projectId]) => localStorage.setItem(key, projectId), [SELECTED_PROJECT_KEY, fixture.projectId]);
  const renderStarted = performance.now();
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  await expect(page.locator('#project-select')).toHaveValue(fixture.projectId);
  await expect(page.locator('.graph-toolbar')).toContainText('Draft r1');
  await expect(page.locator('.react-flow__node.flow-node-service')).toBeVisible();
  await expect.poll(() => page.locator('.react-flow__node.flow-node-operator:visible').count()).toBeGreaterThan(1);
  const renderReadyMs = performance.now() - renderStarted;
  const interactionSamples = await measureInteractions(page);
  const drag = await measureDrag(page);
  const devtools = await page.context().newCDPSession(page);
  await devtools.send('HeapProfiler.collectGarbage');
  await devtools.detach();
  const browserStats = await page.evaluate(() => {
    const memory = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
    return {
      renderedNodeCount: document.querySelectorAll('.react-flow__node').length,
      renderedEdgeCount: document.querySelectorAll('.react-flow__edge').length,
      usedJsHeapMiB: memory === undefined ? null : memory.usedJSHeapSize / 1048576,
    };
  });
  return {
    nodeCount,
    edgeCount,
    nodeProfile: fixture.nodeProfile,
    visiblePortRows: fixture.visiblePortRows,
    patchMs: fixture.patchMs,
    renderReadyMs,
    ...browserStats,
    interactionP95Ms: percentile(interactionSamples, 0.95),
    dragCommitMs: drag.commitMs,
    dragAverageFps: drag.averageFps,
    dragFrameP95Ms: drag.frameP95Ms,
  };
}

test('meets the P4 complex graph budget and records the stress curve', async ({ page, request }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const scenarios: GraphScenarioResult[] = [];
  for (const [nodeCount, edgeCount] of [[300, 600], [1000, 2000]] as const) {
    const fixture = await createGraphFixture(request, nodeCount, edgeCount);
    console.log(`P4 graph fixture ready: nodes=${nodeCount} edges=${edgeCount} patchMs=${fixture.patchMs.toFixed(1)}`);
    const scenario = await benchmarkScenario(page, fixture, nodeCount, edgeCount);
    scenarios.push(scenario);
    console.log(`P4 graph scenario ready: ${JSON.stringify(scenario)}`);
  }
  const environment = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    viewport: { width: window.innerWidth, height: window.innerHeight },
  }));
  const evidence = {
    schemaVersion: 'f8studio-graph-performance/1',
    generatedAt: new Date().toISOString(),
    environment,
    budgets: {
      standardNodeCount: 300,
      standardEdgeCount: 600,
      dragAverageFpsMinimum: 50,
      interactionP95MsMaximum: 100,
      stressNodeCount: 1000,
      stressEdgeCount: 2000,
    },
    scenarios,
    pageErrors,
  };
  await writeFile(EVIDENCE_PATH, `${JSON.stringify(evidence, null, 2)}\n`, 'utf8');
  console.log(`P4 graph performance: ${JSON.stringify(scenarios)}`);
  expect(pageErrors).toEqual([]);
  const standard = scenarios[0]!;
  expect(standard.renderedNodeCount).toBeLessThan(standard.nodeCount);
  expect(standard.dragAverageFps).toBeGreaterThanOrEqual(50);
  expect(standard.interactionP95Ms).toBeLessThanOrEqual(100);
});
