import { defineConfig } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const externalServer = process.env.F8STUDIO_E2E_EXTERNAL_SERVER === '1';
const baseURL = process.env.F8STUDIO_E2E_BASE_URL ?? 'http://127.0.0.1:8240';

export default defineConfig({
  testDir: './e2e',
  testIgnore: '**/graph-performance.spec.ts',
  outputDir: './test-results',
  timeout: 60_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['line']],
  use: {
    baseURL,
    browserName: 'chromium',
    channel: 'chrome',
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 900 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
  webServer: externalServer ? undefined : {
    command: 'pixi run -e web-studio python -m f8studio_server --host 127.0.0.1 --port 8240 --web-dist packages/f8studio_web/dist',
    cwd: '../..',
    url: `${baseURL}/api/health`,
    timeout: 30_000,
    reuseExistingServer: false,
    env: { F8STUDIO_DATA_DIR: join(tmpdir(), 'f8studio-web-e2e') },
  },
});
