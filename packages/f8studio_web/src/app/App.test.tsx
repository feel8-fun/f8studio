import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { App } from './App';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.history.replaceState(null, '', '/');
});

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((input: string | URL | Request) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : new URL(input.url).pathname;
      if (path === '/api/projects' || path === '/api/presentation') return Promise.resolve(new Response('[]', { status: 200 }));
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

});

test('shows the connected server version after validating health', async () => {
  render(<App />);

  expect(await screen.findByText('Local server 0.1.0')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Graph Editor' })).toBeInTheDocument();
  expect(screen.getAllByRole('button', { name: 'New project' })).not.toHaveLength(0);
});

test.each(['video', 'audio', 'three'])('retired %s workspace falls back to Graph without media lab navigation', async (view) => {
  window.history.replaceState(null, '', `/?view=${view}`);
  render(<App />);
  expect(await screen.findByText('Local server 0.1.0')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Graph Editor' })).toBeInTheDocument();
  const navigation = within(screen.getByRole('complementary', { name: 'Workspace navigation' }));
  expect(navigation.getByRole('button', { name: 'Outputs' })).toBeInTheDocument();
  for (const label of ['Video', 'Audio', '3D']) {
    expect(navigation.queryByRole('button', { name: label })).not.toBeInTheDocument();
  }
  expect(screen.queryByRole('tablist', { name: 'Media view' })).not.toBeInTheDocument();
});
