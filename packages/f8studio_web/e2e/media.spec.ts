import { mockPresentation } from './presentationFixture';
import { expect, test, type Locator, type Page } from '@playwright/test';

async function canvasScreenshotSignature(page: Page, canvas: Locator): Promise<{ readonly visiblePixels: number; readonly sum: number }> {
  const encoded = (await canvas.screenshot()).toString('base64');
  return page.evaluate(async (base64) => {
    const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const sample = document.createElement('canvas');
    sample.width = bitmap.width;
    sample.height = bitmap.height;
    const context = sample.getContext('2d');
    if (context === null) throw new Error('2D canvas context is unavailable');
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
    let visiblePixels = 0;
    let sum = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const intensity = (pixels[index] ?? 0) + (pixels[index + 1] ?? 0) + (pixels[index + 2] ?? 0);
      sum += intensity;
      if (intensity > 75) visiblePixels += 1;
    }
    return { visiblePixels, sum };
  }, encoded);
}

async function openVideoOutput(page: Page, source = 'synthetic://bars'): Promise<void> {
  await mockPresentation(page, [{
    nodeId: 'video-e2e', command: 'viz.video.set', tsMs: Date.now(),
    payload: { videoStreamKey: source, scaleMode: 'fit' },
  }]);
  await page.goto('/?view=outputs&node=video-e2e');
}

test('renders and orbits the Ayva OSR model for a TCode output', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const tsMs = Date.now();
  await mockPresentation(page, [{
    nodeId: 'tcode-e2e', command: 'viz.tcode.snapshot', tsMs,
    payload: { model: 'SR6', line: 'L07500I100 R05000I100\n', channels: { L0: 7500, R0: 5000 }, resetVersion: 0 },
  }]);
  await page.goto('/?view=outputs&node=tcode-e2e');
  const canvas = page.getByLabel('SR6 3D TCode visualizer').locator('canvas');
  await expect(canvas).toBeVisible();
  await expect(page.getByText('L07500I100 R05000I100')).toBeVisible();
  await expect(page.getByText('I1', { exact: true })).toHaveCount(0);
  await expect.poll(async () => (await canvasScreenshotSignature(page, canvas)).visiblePixels).toBeGreaterThan(500);

  const before = await canvasScreenshotSignature(page, canvas);
  const bounds = await canvas.boundingBox();
  if (bounds === null) throw new Error('TCode canvas does not have layout bounds');
  await page.mouse.move(bounds.x + bounds.width * 0.65, bounds.y + bounds.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 0.35, bounds.y + bounds.height * 0.4, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => (await canvasScreenshotSignature(page, canvas)).sum).not.toBe(before.sum);
  await page.screenshot({ path: testInfo.outputPath('tcode.png'), fullPage: true });
  expect(pageErrors).toEqual([]);
});

test('shows retained TCode channels in Bars mode without creating a 3D canvas', async ({ page }) => {
  const tsMs = Date.now();
  await mockPresentation(page, [{
    nodeId: 'tcode-bars-e2e', command: 'viz.tcode.snapshot', tsMs,
    payload: { model: 'SR6', line: 'R09999I500\n', channels: { L0: 5000, R0: 9999 }, resetVersion: 0 },
  }]);
  await page.goto('/?view=outputs&node=tcode-bars-e2e');
  await page.getByRole('button', { name: 'Show channel bars' }).click();
  await expect(page.locator('.tcode-channels-primary')).toBeVisible();
  await expect(page.locator('.tcode-channels-primary')).toContainText('L0');
  await expect(page.locator('.tcode-channels-primary')).toContainText('5000');
  await expect(page.locator('.tcode-channels-primary')).toContainText('R0');
  await expect(page.locator('.tcode-channels-primary')).toContainText('9999');
  await expect(page.locator('.tcode-stage canvas')).toHaveCount(0);
  await page.reload();
  await expect(page.locator('.tcode-channels-primary')).toBeVisible();
});

async function leaveOutput(page: Page): Promise<void> {
  await page.getByRole('complementary', { name: 'Workspace navigation' })
    .getByRole('button', { name: 'Logs', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Log Center' })).toBeVisible();
}

async function videoSignature(page: Page): Promise<number[]> {
  return page.locator('video').evaluate((video: HTMLVideoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 18;
    const context = canvas.getContext('2d');
    if (context === null) throw new Error('2D canvas context is unavailable');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const buckets = [0, 0, 0, 0];
    for (let index = 0; index < pixels.length; index += 4) {
      const bucket = Math.floor(index / 4) % buckets.length;
      buckets[bucket] = (buckets[bucket] ?? 0) + (pixels[index] ?? 0) + (pixels[index + 1] ?? 0) + (pixels[index + 2] ?? 0);
    }
    return buckets;
  });
}

async function webGlSignature(page: Page): Promise<{ readonly sum: number; readonly visiblePixels: number }> {
  return page.getByTestId('three-stage').locator('canvas').evaluate(async (canvas: HTMLCanvasElement) => {
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    const context = canvas.getContext('webgl2');
    if (context === null) throw new Error('WebGL2 context is unavailable');
    const width = canvas.width;
    const height = canvas.height;
    const pixels = new Uint8Array(width * height * 4);
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let sum = 0;
    let visiblePixels = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const intensity = (pixels[index] ?? 0) + (pixels[index + 1] ?? 0) + (pixels[index + 2] ?? 0);
      sum += intensity;
      if (intensity > 70) visiblePixels += 1;
    }
    return { sum, visiblePixels };
  });
}

test('plays changing video through the focused Video Viz output', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await openVideoOutput(page);
  await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => ({
    ready: video.readyState, width: video.videoWidth, height: video.videoHeight,
  }))).toMatchObject({ ready: 4, width: 640, height: 360 });
  const firstVideo = await videoSignature(page);
  await expect.poll(() => videoSignature(page)).not.toEqual(firstVideo);
  await page.screenshot({ path: testInfo.outputPath('video.png'), fullPage: true });
  await leaveOutput(page);
  expect(pageErrors).toEqual([]);
});

test('renders an interactive focused 3D output without overflow', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await mockPresentation(page, [{
    nodeId: 'three-e2e', command: 'viz.three_d.set', tsMs: Date.now(),
    payload: { tsMs: Date.now(), worldUp: '+y', people: [{
      name: 'Test', bbox: null, skeletonProtocol: 'test', skeletonEdges: [[0, 1], [1, 2]],
      nodes: [
        { index: 0, name: 'Root', pos: [0, 0, 0], rot: null },
        { index: 1, name: 'Head', pos: [0, 2, 0], rot: null },
        { index: 2, name: 'Hand', pos: [1, 1, 0.5], rot: null },
      ],
    }] },
  }]);
  await page.goto('/?view=outputs&node=three-e2e');
  const canvas = page.getByTestId('three-stage').locator('canvas');
  await expect(canvas).toBeVisible();
  await expect(page.locator('.scene-hud')).toContainText('3 joints');
  await expect.poll(async () => (await webGlSignature(page)).visiblePixels).toBeGreaterThan(1_000);
  const beforeOrbit = await webGlSignature(page);
  const bounds = await canvas.boundingBox();
  if (bounds === null) throw new Error('3D canvas does not have layout bounds');
  expect(bounds.height).toBeGreaterThan(500);
  await page.mouse.move(bounds.x + bounds.width * 0.7, bounds.y + bounds.height * 0.45);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 0.3, bounds.y + bounds.height * 0.35, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => (await webGlSignature(page)).sum).not.toBe(beforeOrbit.sum);
  await page.screenshot({ path: testInfo.outputPath('three.png'), fullPage: true });
  const layout = await page.evaluate(() => ({
    viewportWidth: window.innerWidth, documentWidth: document.documentElement.scrollWidth,
    viewportHeight: window.innerHeight, documentHeight: document.documentElement.scrollHeight,
  }));
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
  expect(layout.documentHeight).toBeLessThanOrEqual(layout.viewportHeight);
  await page.getByRole('complementary', { name: 'Workspace navigation' })
    .getByRole('button', { name: 'Outputs', exact: true }).click();
  await expect(page.locator('.scene-hud')).toHaveCount(0);
  await expect.poll(() => canvas.evaluate((element: HTMLCanvasElement) => {
    const context = element.getContext('webgl2');
    if (context === null) throw new Error('WebGL2 context is unavailable');
    const pixels = new Uint8Array(element.width * element.height * 4);
    context.readPixels(0, 0, element.width, element.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let joints = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index]! > 150 && pixels[index + 1]! > 100 && pixels[index + 2]! < 130) joints += 1;
    }
    return joints;
  })).toBeGreaterThan(10);
  expect(pageErrors).toEqual([]);
});

test('3D outputs wait for real skeleton data instead of showing a demo', async ({ page }) => {
  await mockPresentation(page, [{
    nodeId: 'three-empty-e2e', command: 'viz.three_d.world_up', tsMs: Date.now(),
    payload: { worldUp: '+y' },
  }]);
  await page.goto('/?view=outputs&node=three-empty-e2e');
  await expect(page.getByText('Waiting for skeleton')).toBeVisible();
  await expect(page.getByTestId('three-stage')).toHaveCount(0);
});

test('Audio Viz renders a live waveform and keeps browser listening off by default', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await mockPresentation(page, [{
    nodeId: 'audio-e2e',
    command: 'viz.audio.set',
    payload: { audioStreamKey: 'synthetic://tone', historyMs: 250, throttleMs: 20, channel: 0 },
    tsMs: Date.now(),
  }]);
  await page.goto('/?view=outputs&node=audio-e2e');
  const panel = page.locator('.presentation-audio');
  await expect(panel).toBeVisible();
  await expect(panel.getByRole('button', { name: 'Listen in browser' })).toHaveAttribute('aria-pressed', 'false');
  await expect(panel.getByRole('status')).toContainText(/Live|start preview/);
  await panel.getByRole('tab', { name: 'Wave' }).click();
  await expect(panel.getByRole('status')).toHaveText('Live');
  await expect(panel.getByRole('button', { name: 'Listen in browser' })).toHaveAttribute('aria-pressed', 'false');
  const canvas = panel.getByTestId('audio-viz-canvas');
  await expect.poll(() => canvas.evaluate((element: HTMLCanvasElement) => {
    const context = element.getContext('2d');
    if (context === null) return false;
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let minY = element.height;
    let maxY = 0;
    let greenPixels = 0;
    for (let y = 0; y < element.height; y += 1) {
      for (let x = 0; x < element.width; x += 1) {
        const offset = (y * element.width + x) * 4;
        const red = pixels[offset] ?? 0;
        const green = pixels[offset + 1] ?? 0;
        const blue = pixels[offset + 2] ?? 0;
        if (green > red + 20 && green > blue + 20) {
          greenPixels += 1;
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }
    return greenPixels > 100 && maxY - minY > 8;
  })).toBe(true);
  await panel.getByRole('tab', { name: 'Spectrum' }).click();
  await expect(panel.getByRole('tab', { name: 'Spectrum' })).toHaveAttribute('aria-selected', 'true');
  await panel.getByRole('button', { name: 'Listen in browser' }).click();
  await expect(panel.getByRole('button', { name: 'Mute browser audio' })).toHaveAttribute('aria-pressed', 'true');
  expect(pageErrors).toEqual([]);
});

test('Audio Viz reports no signal when its source has no audio chunks', async ({ page }) => {
  await mockPresentation(page, [{
    nodeId: 'audio-silent-e2e',
    command: 'viz.audio.set',
    payload: { audioStreamKey: 'f8/svc/missing/nodes/missing/data/audio', channel: 0 },
    tsMs: Date.now(),
  }]);
  await page.goto('/?view=outputs&node=audio-silent-e2e');
  const panel = page.locator('.presentation-audio');
  await expect(panel.getByRole('status')).toHaveText('No audio signal');
  await expect(panel.getByRole('button', { name: 'Listen in browser' })).toHaveAttribute('aria-pressed', 'false');
});

test('leaving an output during video negotiation releases the remote session', async ({ page }) => {
  let markRequestStarted: (() => void) | undefined;
  const requestStarted = new Promise<void>((resolve) => { markRequestStarted = resolve; });
  let createdSessionId = '';
  await page.route('**/api/media/sessions', async (route) => {
    const response = await route.fetch();
    const body: { sessionId: string } = await response.json();
    createdSessionId = body.sessionId;
    markRequestStarted?.();
    // Keep the answer pending beyond the pool's 1.5 s navigation grace period.
    await new Promise<void>((resolve) => setTimeout(resolve, 2500));
    await route.fulfill({ response });
  });
  await openVideoOutput(page);
  await requestStarted;
  const released = page.waitForResponse((response) => response.request().method() === 'DELETE' &&
    new URL(response.url()).pathname === `/api/media/sessions/${createdSessionId}`);
  await leaveOutput(page);
  expect((await released).ok()).toBe(true);
});

test('measures Video Viz latency from a 1080p source capture marker', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', '1080p latency is measured once on the desktop viewport');
  await openVideoOutput(page, 'synthetic://bars-1080p');
  await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => ({
    ready: video.readyState,
    width: video.videoWidth,
    height: video.videoHeight,
  }))).toMatchObject({ ready: 4, width: 640, height: 360 });

  const samples = await page.locator('.presentation-video').evaluate(async (stage) => {
    const video = stage.querySelector('video');
    if (video === null) throw new Error('Video element is unavailable');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (context === null) throw new Error('2D context is unavailable');
    const output: number[] = [];
    const deadline = performance.now() + 12_000;
    while (output.length < 60 && performance.now() < deadline) {
      await new Promise<void>((resolve) => {
        video.requestVideoFrameCallback(() => resolve());
      });
      context.drawImage(video, 0, 0);
      let timestampModulo = 0;
      for (let bitIndex = 0; bitIndex < 16; bitIndex += 1) {
        const pixel = context.getImageData(Math.floor((bitIndex * 16 + 8) * video.videoWidth / 1920), Math.floor(8 * video.videoHeight / 1080), 1, 1).data;
        if ((pixel[0] ?? 0) > 128) timestampModulo |= 1 << bitIndex;
      }
      const now = Date.now();
      const base = now - (now & 0xFFFF);
      let capturedAt = base + timestampModulo;
      if (capturedAt > now + 32_768) capturedAt -= 65_536;
      if (capturedAt < now - 32_768) capturedAt += 65_536;
      const latency = now - capturedAt;
      if (latency >= 0 && latency < 5_000) output.push(latency);
    }
    return output;
  });
  expect(samples.length).toBeGreaterThanOrEqual(50);
  const sorted = [...samples].sort((left, right) => left - right);
  const p95 = sorted[Math.ceil(sorted.length * 0.95) - 1] ?? Number.POSITIVE_INFINITY;
  console.log(`Video Viz latency (1080p source): samples=${samples.length} p95=${p95}ms`);
  expect(p95).toBeLessThanOrEqual(200);
  await leaveOutput(page);
});
