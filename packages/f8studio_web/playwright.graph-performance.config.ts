import { defineConfig } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const baseURL = 'http://127.0.0.1:8242';

export default defineConfig({
  testDir: './e2e',
  testMatch: '**/graph-performance.spec.ts',
  outputDir: './test-results/graph-performance',
  timeout: 180_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['line']],
  use: {
    baseURL,
    browserName: 'chromium',
    channel: 'chrome',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'graph-performance' }],
  webServer: {
    command: 'pixi run -e web-studio python -m f8studio_server --host 127.0.0.1 --port 8242 --web-dist packages/f8studio_web/dist',
    cwd: '../..',
    url: `${baseURL}/api/health`,
    timeout: 30_000,
    reuseExistingServer: false,
    env: { F8STUDIO_DATA_DIR: join(tmpdir(), 'f8studio-graph-performance') },
  },
});
