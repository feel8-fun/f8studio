// Integration regression: delayed responses must stay on their original page.
const { readFileSync, existsSync } = require('node:fs');
const { resolve } = require('node:path');
const assert = require('node:assert/strict');
const { chromium, expect } = require('../../extensions/f8webstudio/f8studio_web/node_modules/@playwright/test');

const assets = resolve(__dirname, '../../platform/f8platform/portal_assets');
const instrument = `
  window.portalLoads=[];
  const originalLoad=load;
  load=(...args)=>{const pending=originalLoad(...args);window.portalLoads.push(pending);return pending;};
  window.portalDetails=[];
  const originalInspect=inspect;
  inspect=(...args)=>{const pending=originalInspect(...args);window.portalDetails.push(pending);return pending;};
`;
const storage = { path: '/storage', cachePath: '/cache', totalUsage: {}, environmentUsage: {}, cacheUsage: {},
  unusedEnvironments: [] };
const tool = name => ({ extensionId: 'debug', toolId: 'inspect', name, description: 'Inspect a stream', fields: [] });

function gate() {
  let enter, release;
  const entered = new Promise(resolve => { enter = resolve; });
  const wait = new Promise(resolve => { release = resolve; });
  return { entered, enter, wait, release };
}

async function checkNavigation(browser, ignoreAbort) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  let delayed = null;
  let toolName = 'Inspect stream';
  let jobs = [];
  await page.addInitScript(({ ignoreAbort }) => {
    if (!ignoreAbort) return;
    const originalFetch = window.fetch.bind(window);
    window.fetch = (url, options) => originalFetch(url, { ...options, signal: undefined });
  }, { ignoreAbort });
  await page.route('http://platform.test/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) {
      const file = path === '/' ? 'index.html' : path.slice(1);
      const source = readFileSync(resolve(assets, file), 'utf8');
      await route.fulfill({ contentType: file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html',
        body: source + (file === 'portal.js' ? instrument : '') });
      return;
    }
    let body = path === '/api/environments/storage' ? storage
      : path === '/api/startup' ? { applications: [] }
      : path === '/api/extension-tools' ? [tool(toolName)]
      : path === '/api/tool-jobs' ? jobs
      : path === '/api/tool-jobs/job-one' ? { jobId: 'job-one', status: 'running', stdout: 'Tool output' }
      : path === '/api/service-processes/logs' ? [{ serviceId: 'process-one', line: 'Process output' }] : [];
    let status = 200;
    const pending = delayed;
    if (pending && pending.path === path) {
      delayed = null;
      body = pending.body ?? body;
      status = pending.status ?? status;
      pending.enter();
      await pending.wait;
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
  try {
    await page.goto('http://platform.test/');
    await expect(page.locator('#content h1')).toHaveText('Extensions');
    await page.getByRole('button', { name: 'Runtime Environments', exact: true }).click();
    await expect(page.locator('#content h1')).toHaveText('Runtime Environments');
    await expect(page.getByRole('button', { name: 'Clear package cache', exact: true })).toHaveCount(0);
    for (const destination of ['Tools', 'Processes']) {
      const pending = gate();
      delayed = { ...pending, path: '/api/environments/storage' };
      await page.getByRole('button', { name: 'Runtime Environments', exact: true }).click();
      await pending.entered;
      await expect(page.locator('#content h1')).toHaveText('Runtime Environments');
      await page.getByRole('button', { name: destination, exact: true }).click();
      const title = destination === 'Tools' ? 'Tools' : 'Service processes';
      await expect(page.locator('#content h1')).toHaveText(title);
      pending.release();
      await page.evaluate(() => Promise.allSettled(window.portalLoads));
      await expect(page.locator('#content h1')).toHaveText(title);
      await expect(page.locator('#content')).not.toContainText('Storage and package cache');
    }

    const failed = gate();
    delayed = { ...failed, path: '/api/environments/storage', status: 503, body: { detail: 'Old storage error' } };
    await page.getByRole('button', { name: 'Runtime Environments', exact: true }).click();
    await failed.entered;
    await page.getByRole('button', { name: 'Processes', exact: true }).click();
    failed.release();
    await page.evaluate(() => Promise.allSettled(window.portalLoads));
    await expect(page.locator('#content h1')).toHaveText('Service processes');
    await expect(page.locator('#error')).toBeHidden();

    const obsolete = gate();
    delayed = { ...obsolete, path: '/api/extension-tools', body: [tool('Obsolete tool')] };
    await page.getByRole('button', { name: 'Tools', exact: true }).click();
    await obsolete.entered;
    toolName = 'Latest tool';
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Latest tool', exact: true })).toBeVisible();
    obsolete.release();
    await page.evaluate(() => Promise.allSettled(window.portalLoads));
    await expect(page.locator('#content h1')).toHaveText('Tools');
    await expect(page.getByRole('heading', { name: 'Latest tool', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Obsolete tool', exact: true })).toHaveCount(0);

    jobs = [{ jobId: 'job-one', extensionId: 'debug', toolId: 'inspect', status: 'running' }];
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await page.locator('.tool-history summary').click();
    await expect(page.getByRole('button', { name: 'Result / logs', exact: true })).toBeVisible();
    const oldDetail = gate();
    delayed = { ...oldDetail, path: '/api/tool-jobs/job-one' };
    await page.getByRole('button', { name: 'Result / logs', exact: true }).click();
    await oldDetail.entered;
    await page.getByRole('button', { name: 'Open tool', exact: true }).click();
    oldDetail.release();
    await page.evaluate(() => Promise.allSettled(window.portalDetails));
    await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible();
    await expect(page.locator('#detail h2')).toHaveText('Latest tool');

    await page.getByRole('button', { name: 'Result / logs', exact: true }).click();
    await expect(page.locator('#detail pre')).toContainText('Tool output');
    const oldPoll = gate();
    delayed = { ...oldPoll, path: '/api/tool-jobs/job-one' };
    await oldPoll.entered;
    await page.getByRole('button', { name: 'Processes', exact: true }).click();
    await page.getByRole('button', { name: 'View logs', exact: true }).click();
    await expect(page.locator('#detail pre')).toHaveText('process-one: Process output');
    oldPoll.release();
    await page.waitForFunction('pollingJob === false');
    await expect(page.locator('#detail pre')).toHaveText('process-one: Process output');
    assert.deepEqual(errors, []);
  } finally {
    delayed?.release();
    await page.close();
  }
}

async function checkTasksAndUrls(browser) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const extension = id => ({ extensionId: id, name: id, description: 'Fixture', version: '1.0', state: 'available',
    serviceClasses: [], toolIds: [], skillIds: [] });
  const items = [extension('first'), extension('second')];
  const environment = { environmentId: 'runtime', name: 'Fixture runtime', state: 'missing', ready: false,
    runtimeKind: 'pixi', detail: '', extensionIds: ['first'], serviceClasses: [], canRemove: false };
  let jobs = [];
  let usage = 1024 * 1024;
  let version = '1.0';
  let storageReads = 0;
  let log = 'Installer is downloading packages';
  await page.route('http://platform.test/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) {
      const file = path === '/' ? 'index.html' : path.slice(1);
      await route.fulfill({ contentType: file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html',
        body: readFileSync(resolve(assets, file), 'utf8') });
      return;
    }
    let status = 200;
    let body = [];
    if (route.request().method() === 'POST' && (path.endsWith('/install') || path.endsWith('/prepare'))) {
      const target = path.split('/')[3];
      const action = path.endsWith('/prepare') ? 'prepare-environment' : 'install-extension';
      const request = action === 'prepare-environment' ? { action, environmentId: target } : { action, extensionId: target };
      body = { jobId: target, request, state: jobs.some(job => job.state === 'running') ? 'queued' : 'running',
        detail: 'Downloading packages', createdAt: Date.now(), cancellable: true };
      jobs.push(body);
      if (action === 'prepare-environment') environment.state = 'preparing';
      status = 202;
    } else if (path === '/api/management-jobs') body = jobs;
    else if (path.endsWith('/logs')) body = { log };
    else if (path === '/api/extensions') body = items;
    else if (path === '/api/startup') body = { applications: [] };
    else if (path === '/api/environments') body = [environment];
    else if (path === '/api/environments/storage') {
      storageReads++;
      body = { ...storage, totalUsage: { uniqueFileBytes: usage }, environmentUsage: { uniqueFileBytes: usage } };
    } else if (path === '/api/environments/runtime/detail') body = { packages: [{ name: 'numpy', version }] };
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
  try {
    await page.goto('http://platform.test/?view=environments');
    await expect(page.locator('#content h1')).toHaveText('Runtime Environments');
    await page.reload();
    await expect(page.locator('#content h1')).toHaveText('Runtime Environments');
    const reads = storageReads;
    usage = 3 * 1024 * 1024;
    await page.getByRole('button', { name: 'Packages and details' }).click();
    await expect(page.locator('#detail pre')).toContainText('1.0');
    version = '2.0';
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await expect(page.locator('#content')).toContainText('file data (hardlinks counted once): 3.0 MiB');
    await expect(page.locator('#detail pre')).toContainText('2.0');
    assert(storageReads > reads);
    await expect(page.locator('#refresh-status')).toContainText('Updated');

    await page.getByRole('button', { name: 'Extensions', exact: true }).click();
    await expect(page).toHaveURL('http://platform.test/?view=extensions');
    const first = page.locator('#content .card').filter({ has: page.getByRole('heading', { name: 'first', exact: true }) });
    const second = page.locator('#content .card').filter({ has: page.getByRole('heading', { name: 'second', exact: true }) });
    await first.getByRole('button', { name: 'Install', exact: true }).click();
    await second.getByRole('button', { name: 'Install', exact: true }).click();
    await expect(page.locator('#tasks')).toContainText('queued');
    items[0].state = 'installed';
    jobs[0] = { ...jobs[0], state: 'succeeded', detail: 'Task completed', cancellable: false };
    jobs[1] = { ...jobs[1], state: 'running' };
    await expect(first.getByRole('button', { name: 'Uninstall', exact: true })).toBeVisible();
    jobs[1] = { ...jobs[1], state: 'failed', detail: 'Dependency resolution failed', cancellable: false };
    await expect(page.locator('#tasks')).toContainText('failed');
    await expect(page.locator('#tasks .task-row').first()).toBeHidden();
    await page.locator('#tasks summary').click();
    await expect(page.locator('#tasks .task-row').first()).toBeVisible();
    await page.getByRole('button', { name: 'Close tasks', exact: true }).click();
    await expect(page.locator('#tasks .task-row')).toHaveCount(0);
    await page.getByRole('button', { name: /Show maintenance tasks/ }).click();

    await page.getByRole('button', { name: 'Runtime Environments', exact: true }).click();
    await page.getByRole('button', { name: 'Packages and details' }).click();
    await page.getByRole('button', { name: 'Verify / prepare' }).click();
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeVisible();
    version = '3.0'; usage = 6 * 1024 * 1024; environment.state = 'ready'; environment.ready = true;
    jobs[jobs.length - 1] = { ...jobs[jobs.length - 1], state: 'succeeded', detail: 'Task completed', cancellable: false };
    await expect(page.locator('#content .badge.ready')).toHaveText('ready');
    await expect(page.locator('#detail pre')).toContainText('3.0');
    await expect(page.locator('#content')).toContainText('file data (hardlinks counted once): 6.0 MiB');

    for (const [name, view, title] of [['Tools', 'tools', 'Tools'], ['Processes', 'processes', 'Service processes'], ['Tasks', 'tasks', 'Tasks']]) {
      await page.getByRole('button', { name, exact: true }).click();
      await expect(page).toHaveURL(`http://platform.test/?view=${view}`);
      await page.reload();
      await expect(page.locator('#content h1')).toHaveText(title);
    }
    await page.goBack();
    await expect(page.locator('#content h1')).toHaveText('Service processes');
    await page.goForward();
    await expect(page.locator('#content h1')).toHaveText('Tasks');
    log = 'Dependency resolution failed: package conflict';
    await page.locator('#content .task-row').filter({ hasText: 'second' }).getByRole('button', { name: 'Task details / logs' }).click();
    await expect(page.locator('#detail pre')).toContainText('package conflict');
    await page.getByRole('button', { name: 'Close details', exact: true }).click();
    await expect(page.locator('#detail')).toBeHidden();
    await page.getByRole('button', { name: 'Clear completed', exact: true }).click();
    await expect(page.locator('#content .task-row')).toHaveCount(0);
    await page.reload();
    await expect(page.locator('#content h1')).toHaveText('Tasks');
    await expect(page.locator('#content .task-row')).toHaveCount(0);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
}

(async () => {
  const executablePath = process.env.F8_CHROMIUM_EXECUTABLE || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);
  const browser = await chromium.launch({ headless: true, executablePath });
  try {
    await checkNavigation(browser, false);
    await checkNavigation(browser, true);
    await checkTasksAndUrls(browser);
    console.log('PASS: navigation races, queued task completion, failures, environment refresh and page URLs');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
