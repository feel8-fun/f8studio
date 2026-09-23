import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthCallbackRoute } from './auth-callback.jsx';

function renderRoute(entry) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AuthCallbackRoute />
    </MemoryRouter>,
  );
}

describe('AuthCallbackRoute', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('renders a completion page and schedules a redirect back to the root portal', () => {
    const setTimeoutSpy = vi.spyOn(window, 'setTimeout');

    renderRoute('/auth-callback?status=success');

    expect(screen.getByText('Desktop sign-in complete')).toBeTruthy();
    expect(screen.getByText('Feel8 Studio should already be completing the sign-in flow in the background.')).toBeTruthy();
    expect(screen.getByText('This page will return to the portal automatically in a moment.')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Open Portal' }).getAttribute('href')).toBe('/login');
    expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 2500);
  });

  it('renders a helpful error page and does not schedule a redirect', () => {
    const setTimeoutSpy = vi.spyOn(window, 'setTimeout');

    renderRoute('/auth-callback?status=error&error=access_denied&error_description=User%20cancelled');

    expect(screen.getByText('Desktop sign-in needs attention')).toBeTruthy();
    expect(screen.getByText('The browser sign-in did not finish cleanly. Return to Feel8 Studio and try again if needed.')).toBeTruthy();
    expect(screen.getByText('access_denied: User cancelled')).toBeTruthy();
    expect(setTimeoutSpy).not.toHaveBeenCalled();
  });
});
