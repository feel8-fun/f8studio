import { afterEach, expect, test, vi } from 'vitest';
import { EventStream } from './eventStream';

class Socket {
  static instances: Socket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(readonly url: string) { Socket.instances.push(this); }
  close() { this.onclose?.(); }
  send(value: unknown) { this.onmessage?.({ data: JSON.stringify(value) }); }
}

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); Socket.instances = []; });

test('shares one connection and resumes from processed cursor, not hello head', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  const stream = new EventStream();
  const events: number[] = [];
  const resync = vi.fn();
  const stopA = stream.subscribe((event) => events.push(event.sequence), resync);
  const stopB = stream.subscribe(() => {});
  expect(Socket.instances).toHaveLength(1);
  const first = Socket.instances[0]!;
  first.send({ type: 'stream.hello', serverEpoch: 'epoch', sequence: 0, resumed: false });
  first.send({ type: 'graph.committed', serverEpoch: 'epoch', sequence: 1, scope: 'project:p', payload: {} });
  first.close();
  await vi.advanceTimersByTimeAsync(250);
  const second = Socket.instances[1]!;
  expect(second.url).toContain('epoch=epoch&after=1');
  second.send({ type: 'stream.hello', serverEpoch: 'epoch', sequence: 3, resumed: true });
  second.send({ type: 'graph.committed', serverEpoch: 'epoch', sequence: 2, scope: 'project:p', payload: {} });
  second.send({ type: 'graph.committed', serverEpoch: 'epoch', sequence: 3, scope: 'project:p', payload: {} });
  expect(events).toEqual([1, 2, 3]);
  stopA(); stopB();
});

test('consumer failures do not block other consumers or replay delivered events', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  const report = vi.spyOn(console, 'error').mockImplementation(() => {});
  const stream = new EventStream();
  const fail = () => { throw new Error('broken projection'); };
  const stopA = stream.subscribe(fail, fail, fail);
  const event = vi.fn();
  const resync = vi.fn();
  const connection = vi.fn();
  const stopB = stream.subscribe(event, resync, connection);
  const first = Socket.instances[0]!;
  first.onopen?.();
  first.send({ type: 'stream.hello', serverEpoch: 'epoch', sequence: 0, resumed: false });
  first.send({ type: 'graph.committed', serverEpoch: 'epoch', sequence: 1, scope: 'project:p', payload: {} });
  expect(event).toHaveBeenCalledTimes(1);
  expect(resync).toHaveBeenCalled();
  expect(connection).toHaveBeenCalledWith(true);
  expect(Socket.instances).toHaveLength(1);
  first.close();
  expect(connection).toHaveBeenCalledWith(false);
  await vi.advanceTimersByTimeAsync(250);
  expect(Socket.instances[1]!.url).toContain('after=1');
  expect(report).toHaveBeenCalledWith('Studio event consumer failed', expect.any(Error));
  stopA(); stopB();
});

test('retries constructor failures and cancels retries when unsubscribed', async () => {
  vi.useFakeTimers();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.stubGlobal('WebSocket', class { constructor() { throw new Error('unavailable'); } });
  const stream = new EventStream();
  const stop = stream.subscribe(() => {});
  vi.stubGlobal('WebSocket', Socket);
  await vi.advanceTimersByTimeAsync(250);
  expect(Socket.instances).toHaveLength(1);
  Socket.instances[0]!.close();
  stop();
  await vi.advanceTimersByTimeAsync(10_000);
  expect(Socket.instances).toHaveLength(1);
});
