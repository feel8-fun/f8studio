import react from '@vitejs/plugin-react';
import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  build: { outDir: '../../build/web-studio', emptyOutDir: true },
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8210', ws: true },
    },
  },
  test: {
    environment: 'jsdom',
    exclude: [...configDefaults.exclude, 'e2e/**'],
    setupFiles: ['./src/test/setup.ts'],
  },
});
