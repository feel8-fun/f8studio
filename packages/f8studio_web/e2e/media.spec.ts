import { expect, test, type Page } from '@playwright/test';

async function videoSignature(page: Page): Promise<number[]> {
  return page.locator('video').evaluate((video) => {
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
  return page.getByTestId('three-stage').locator('canvas').evaluate(async (canvas) => {
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

test('plays changing WebRTC video and renders interactive 3D without overflow', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();

  await page.getByRole('button', { name: 'Connect' }).click();
  await expect(page.locator('.media-status')).toContainText('fps');
  await expect.poll(() => page.locator('video').evaluate((video) => ({
    ready: video.readyState,
    width: video.videoWidth,
    height: video.videoHeight,
  }))).toMatchObject({ ready: 4, width: 640, height: 360 });
  const firstVideo = await videoSignature(page);
  await page.waitForTimeout(350);
  const secondVideo = await videoSignature(page);
  expect(secondVideo).not.toEqual(firstVideo);
  await page.locator('video').click();
  await expect(page.locator('.media-sample')).toContainText('Latest raw');
  await page.screenshot({ path: testInfo.outputPath('video.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop' }).click();
  await expect(page.locator('.media-status')).toHaveText('Ready');

  await page.getByRole('tab', { name: '3D' }).click();
  const canvas = page.getByTestId('three-stage').locator('canvas');
  await expect(canvas).toBeVisible();
  await expect.poll(async () => (await webGlSignature(page)).visiblePixels).toBeGreaterThan(1_000);
  const beforeOrbit = await webGlSignature(page);
  const bounds = await canvas.boundingBox();
  if (bounds === null) throw new Error('3D canvas does not have layout bounds');
  await page.mouse.move(bounds.x + bounds.width * 0.7, bounds.y + bounds.height * 0.45);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 0.3, bounds.y + bounds.height * 0.35, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const afterOrbit = await webGlSignature(page);
  expect(afterOrbit.sum).not.toBe(beforeOrbit.sum);
  await page.screenshot({ path: testInfo.outputPath('three.png'), fullPage: true });

  const layout = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    viewportHeight: window.innerHeight,
    documentHeight: document.documentElement.scrollHeight,
  }));
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
  expect(layout.documentHeight).toBeLessThanOrEqual(layout.viewportHeight);
  expect(pageErrors).toEqual([]);
});

test('plays WebRTC audio and renders a nonflat waveform', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.connection-online')).toBeVisible();
  await page.getByRole('tab', { name: 'Audio' }).click();
  await page.getByRole('button', { name: 'Play' }).click();
  await expect(page.locator('.media-status')).toContainText('48000 Hz');

  const canvas = page.getByTestId('audio-waveform');
  await expect(canvas).toBeVisible();
  await expect.poll(() => canvas.evaluate((element) => {
    const context = element.getContext('2d');
    if (context === null) throw new Error('2D context is unavailable');
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let greenPixels = 0;
    let minY = element.height;
    let maxY = 0;
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
  const signature = await canvas.evaluate((element) => {
    const context = element.getContext('2d');
    if (context === null) throw new Error('2D context is unavailable');
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let greenPixels = 0;
    let minY = element.height;
    let maxY = 0;
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
    return { greenPixels, amplitude: maxY - minY };
  });
  expect(signature.greenPixels).toBeGreaterThan(100);
  expect(signature.amplitude).toBeGreaterThan(8);
  await page.screenshot({ path: testInfo.outputPath('audio.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop' }).click();
  await expect(page.locator('.media-status')).toHaveText('Ready');
  expect(pageErrors).toEqual([]);
});

test('stopping during video negotiation releases the remote session', async ({ page }) => {
  let markRequestStarted: (() => void) | null = null;
  let markRequestFinished: (() => void) | null = null;
  const requestReachedServer = new Promise<void>((resolve) => {
    markRequestStarted = resolve;
  });
  const requestFinished = new Promise<void>((resolve) => {
    markRequestFinished = resolve;
  });
  await page.route('**/api/media/sessions', async (route) => {
    markRequestStarted?.();
    await new Promise<void>((resolve) => setTimeout(resolve, 300));
    const response = await route.fetch();
    await route.fulfill({ response });
    markRequestFinished?.();
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect' }).click();
  await requestReachedServer;
  await page.getByRole('button', { name: 'Stop' }).click();
  await expect(page.locator('.media-status')).toHaveText('Ready');
  await requestFinished;
  await expect.poll(() => page.evaluate(async () => {
    const response = await fetch('/api/media/metrics');
    const metrics = await response.json() as { videoSessions: number; videoSources: number };
    return [metrics.videoSessions, metrics.videoSources];
  })).toEqual([0, 0]);
});

test('measures 1080p displayed video latency from the capture marker', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', '1080p latency is measured once on the desktop viewport');
  await page.goto('/');
  await page.getByLabel('Video source').fill('synthetic://bars-1080p');
  await page.getByRole('button', { name: 'Main' }).click();
  await page.getByRole('button', { name: 'Connect' }).click();
  await expect.poll(() => page.locator('video').evaluate((video) => ({
    ready: video.readyState,
    width: video.videoWidth,
    height: video.videoHeight,
  }))).toMatchObject({ ready: 4, width: 1920, height: 1080 });

  const samples = await page.locator('.video-stage').evaluate(async (stage) => {
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
        const pixel = context.getImageData(bitIndex * 16 + 8, 8, 1, 1).data;
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
  console.log(`1080p displayed latency: samples=${samples.length} p95=${p95}ms`);
  expect(p95).toBeLessThanOrEqual(200);
  await page.getByRole('button', { name: 'Stop' }).click();
});
