import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
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
      if (path === '/api/projects' || path === '/api/presentation' || path === '/api/extension-tools' || path === '/api/tool-jobs' || path === '/api/extensions' || path === '/api/environments') return Promise.resolve(new Response('[]', { status: 200 }));
      if (path === '/api/environments/storage') return Promise.resolve(new Response(JSON.stringify({ path: '/data', cachePath: '/data/cache', canChange: true }), { status: 200 }));
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

test.each(['services', 'local', 'video', 'audio', 'three'])('retired %s workspace falls back to Graph', async (view) => {
  window.history.replaceState(null, '', `/?view=${view}`);
  render(<App />);
  expect(await screen.findByText('Local server 0.1.0')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Graph Editor' })).toBeInTheDocument();
  const navigation = within(screen.getByRole('complementary', { name: 'Workspace navigation' }));
  expect(navigation.getByRole('button', { name: 'Outputs' })).toBeInTheDocument();
  for (const label of ['Services', 'Local integrations', 'Video', 'Audio', '3D']) {
    expect(navigation.queryByRole('button', { name: label })).not.toBeInTheDocument();
  }
  expect(screen.queryByRole('tablist', { name: 'Media view' })).not.toBeInTheDocument();
});

test('tools URL opens Tools', async () => {
  window.history.replaceState(null, '', '/?view=tools');
  render(<App />);
  expect(await screen.findByRole('navigation', { name: 'Tool catalog' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Tools' })).toBeInTheDocument();
  const navigation = within(screen.getByRole('complementary', { name: 'Workspace navigation' }));
  expect(navigation.getByRole('button', { name: 'Tools' })).toBeInTheDocument();
  expect(navigation.queryByRole('button', { name: 'Local integrations' })).not.toBeInTheDocument();
});

test('extensions URL opens Extensions with runtime management on a separate page', async () => {
  window.history.replaceState(null, '', '/?view=extensions');
  render(<App />);
  await screen.findByRole('textbox', { name: 'Search extensions' });
  expect(screen.getByRole('heading', { name: 'Extensions', level: 1 })).toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Runtime environments' })).not.toBeInTheDocument();
  const navigation = within(screen.getByRole('complementary', { name: 'Workspace navigation' }));
  expect(navigation.queryByRole('button', { name: 'Services' })).not.toBeInTheDocument();
  fireEvent.click(navigation.getByRole('button', { name: 'Runtime Environments' }));
  await screen.findByRole('region', { name: 'Runtime environments' });
  expect(screen.getByRole('heading', { name: 'Runtime Environments', level: 1 })).toBeInTheDocument();
  expect(window.location.search).toBe('?view=environments');
  fireEvent.click(navigation.getByRole('button', { name: 'Extensions' }));
  await screen.findByRole('textbox', { name: 'Search extensions' });
  expect(window.location.search).toBe('?view=extensions');
});

test('opens Runtime Environments directly', async () => {
  window.history.replaceState(null, '', '/?view=environments');
  render(<App />);
  expect(await screen.findByRole('region', { name: 'Runtime environments' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Runtime Environments', level: 1 })).toBeInTheDocument();
});
