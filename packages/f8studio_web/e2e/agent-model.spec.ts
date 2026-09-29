import { expect, test } from '@playwright/test';

// Explicit opt-in: exercises the configured model and incurs provider usage.
test('configured model builds the cosine graph and opens a working approval', async ({ page, request }, testInfo) => {
  test.skip(process.env.F8STUDIO_LIVE_AGENT_TEST !== '1', 'Requires an explicitly enabled live model test');
  test.setTimeout(240_000);
  const projectId = `agent-model-${testInfo.project.name}-${Date.now()}`;
  const providers = await (await request.get('/api/agents/providers')).json() as { providerId: string; models: string[]; configured: boolean }[];
  const provider = providers.find((item) => item.providerId === 'openai' && item.configured);
  expect(provider, 'Configure the OpenAI agent provider before running the live test').toBeDefined();
  expect((await request.post('/api/projects', { data: { projectId, name: 'Live agent cosine test' } })).ok()).toBeTruthy();
  let sessionId = '';
  try {
    const engineId = `${projectId}-engine`;
    const definitions = [
      { kind: 'service', nodeId: engineId, serviceClass: 'f8.pyengine' },
      { kind: 'service', nodeId: 'studio', serviceClass: 'f8.pystudio' },
      { kind: 'operator', nodeId: 'wave', serviceId: 'studio', serviceClass: 'f8.pystudio', operatorClass: 'f8.viz.wave' },
      { kind: 'operator', nodeId: 'tick', serviceId: engineId, serviceClass: 'f8.pyengine', operatorClass: 'f8.tick' },
    ];
    const operations = [];
    for (const definition of definitions) {
      const response = await request.post('/api/catalog/nodes', { data: definition });
      expect(response.ok(), await response.text()).toBeTruthy();
      const node = await response.json() as Record<string, unknown>;
      if (definition.nodeId === 'tick') node.stateValues = { tickMs: 1000 };
      operations.push({ op: 'createNode', node });
    }
    const seeded = await request.post(`/api/projects/${projectId}/patch`, { data: {
      requestId: `${projectId}-seed`, expectedGraphRevision: 0, expectedLayoutRevision: 0, operations,
    } });
    expect(seeded.ok(), await seeded.text()).toBeTruthy();
    const created = await request.post('/api/agents/sessions', { data: {
      projectId, providerId: provider!.providerId, modelId: provider!.models[0], title: 'Cosine regression',
    } });
    expect(created.status()).toBe(201);
    sessionId = (await created.json() as { sessionId: string }).sessionId;
    await page.goto(`/?view=agent&project=${projectId}&session=${sessionId}`);
    await request.post(`/api/agents/sessions/${sessionId}/runs`, { data: {
      prompt: '在pyengine里面搭建一个余弦波发生器, 生成 1hz, 0-1的余弦波, 并且接上tcode L0频道生成tcode, 然后把tcode发送给tcode viz. 生成的余弦波发送给wave viz。只编辑图，本次不要部署或运行。',
    } });
    await expect.poll(async () => (await (await request.get(`/api/agents/sessions/${sessionId}`)).json() as { status: string }).status,
      { timeout: 180_000, intervals: [1000] }).not.toBe('running');
    const waiting = await (await request.get(`/api/agents/sessions/${sessionId}`)).json() as { status: string; messages: unknown; toolCalls: unknown };
    expect(waiting.status, JSON.stringify(waiting)).toBe('waiting_for_approval');
    const approval = page.locator('.agent-approval');
    await expect(approval).toContainText('graph.apply_patch');
    await page.screenshot({ path: testInfo.outputPath('model-approval.png'), fullPage: true });
    await approval.getByRole('button', { name: 'Approve agent tool' }).click();
    await expect(page.locator('.agent-status')).toHaveText('succeeded', { timeout: 60_000 });
    const result = await (await request.get(`/api/projects/${projectId}`)).json() as {
      document: { nodes: { nodeId: string; operatorClass?: string; stateValues: Record<string, unknown>;
        spec: { stateFields: { name: string; valueSchema: { default?: unknown } }[] };
        ports: { portId: string; name: string }[] }[];
        edges: { fromNodeId: string; fromPortId: string; toNodeId: string; toPortId: string }[] };
    };
    const nodes = result.document.nodes;
    const phase = nodes.find((node) => node.operatorClass === 'f8.phase');
    const cosine = nodes.find((node) => node.operatorClass === 'f8.cosine');
    const tcode = nodes.find((node) => node.operatorClass === 'f8.tcode');
    const viz = nodes.find((node) => node.operatorClass === 'f8.viz.tcode');
    expect(phase).toBeDefined(); expect(cosine).toBeDefined(); expect(tcode).toBeDefined(); expect(viz).toBeDefined();
    const state = (node: (typeof nodes)[number], field: string) => node.stateValues[field]
      ?? node.spec.stateFields.find((item) => item.name === field)?.valueSchema.default;
    expect(state(phase!, 'hz')).toBe(1);
    expect(state(cosine!, 'amp')).toBe(0.5);
    expect(state(cosine!, 'dc')).toBe(0.5);
    expect(state(tcode!, 'intervalMs')).toBe(20);
    expect(state(viz!, 'upstreamSampleIntervalMs')).toBe(20);
    expect(state(nodes.find((node) => node.operatorClass === 'f8.viz.wave')!, 'upstreamSampleIntervalMs')).toBe(20);
    const connections = result.document.edges.map((edge) => {
      const source = nodes.find((node) => node.nodeId === edge.fromNodeId)!;
      const target = nodes.find((node) => node.nodeId === edge.toNodeId)!;
      return `${source.operatorClass}.${source.ports.find((port) => port.portId === edge.fromPortId)!.name}->${target.operatorClass}.${target.ports.find((port) => port.portId === edge.toPortId)!.name}`;
    });
    expect(connections).toEqual(expect.arrayContaining([
      'f8.phase.phase->f8.cosine.phase', 'f8.cosine.value->f8.tcode.L0',
      'f8.cosine.value->f8.viz.wave.x', 'f8.tcode.tcode->f8.viz.tcode.tcode',
    ]));
    const session = await (await request.get(`/api/agents/sessions/${sessionId}`)).json() as { toolCalls: { status: string }[] };
    expect(session.toolCalls.every((call) => call.status === 'succeeded')).toBe(true);
    await testInfo.attach('applied-graph', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
    await page.screenshot({ path: testInfo.outputPath('model-complete.png'), fullPage: true });
  } finally {
    if (sessionId) await request.delete(`/api/agents/sessions/${sessionId}/runs/current`);
    expect((await request.delete(`/api/projects/${projectId}`)).status()).toBe(204);
  }
});
