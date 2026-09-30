import type { Page } from '@playwright/test';
import type { LivePatch, LiveSnapshot, PresentationCommand } from '../src/api/contracts.gen';

/** Inject retained presentation data through the same live channel as built-in Viz nodes. */
export async function mockPresentation(page: Page, commands: readonly PresentationCommand[]): Promise<void> {
  const values = Object.fromEntries(commands.map((command, index) => [
    `presentation/${command.nodeId}/${command.command}`, { ...command, seq: index + 1 },
  ]));
  await page.routeWebSocket('**/api/live', (socket) => {
    const server = socket.connectToServer();
    server.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as LiveSnapshot | LivePatch;
      socket.send(JSON.stringify(message.type === 'live.snapshot'
        ? { ...message, values: { ...message.values, ...values } } : message));
    });
  });
}
