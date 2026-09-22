import { render, screen } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';

import { App } from './App';

afterEach(() => {
  vi.restoreAllMocks();
});

test('shows the connected server version after validating health', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((input: string | URL | Request) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : new URL(input.url).pathname;
      if (path === '/api/projects') return Promise.resolve(new Response('[]', { status: 200 }));
      if (path === '/api/catalog') return Promise.resolve(new Response('{"services":[],"operators":[]}', { status: 200 }));
      return Promise.resolve(new Response(
        JSON.stringify({
          status: 'ok',
          service: 'f8studio-server',
          version: '0.1.0',
          protocol_version: 'f8studio-api/1',
          server_epoch: 'epoch-1',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ));
    }),
  );

  render(<App />);

  expect(await screen.findByText('Local server 0.1.0')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Graph Editor' })).toBeInTheDocument();
  expect(screen.getAllByRole('button', { name: 'New project' })).not.toHaveLength(0);
});
