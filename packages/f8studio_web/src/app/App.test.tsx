import { render, screen } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import { App } from './App';

afterEach(() => {
  vi.restoreAllMocks();
});

test('shows the connected server version after validating health', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'ok',
          service: 'f8studio-server',
          version: '0.1.0',
          protocol_version: 'f8studio-api/1',
          server_epoch: 'epoch-1',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    ),
  );

  render(<App />);

  expect(await screen.findByText('Local server 0.1.0')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Media Lab' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Connect' })).toBeInTheDocument();
});
