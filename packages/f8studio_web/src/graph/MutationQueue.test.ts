import { expect, test } from 'vitest';
import { MutationQueue } from './MutationQueue';

test('serializes rapid edits and reads revision when each edit executes', async () => {
  const queue = new MutationQueue();
  let revision = 0;
  let unblock: () => void = () => {};
  const blocked = new Promise<void>((resolve) => { unblock = resolve; });
  const revisions: number[] = [];
  const first = queue.enqueue(async () => { revisions.push(revision); await blocked; revision += 1; });
  const second = queue.enqueue(async () => { revisions.push(revision); revision += 1; });
  await Promise.resolve();
  expect(revisions).toEqual([0]);
  expect(queue.pending).toBe(2);
  unblock();
  await Promise.all([first, second]);
  expect(revisions).toEqual([0, 1]);
  expect(queue.pending).toBe(0);
});

test('reports failures and cancels dependent edits without poisoning future work', async () => {
  const queue = new MutationQueue();
  const first = queue.enqueue(async () => { throw new Error('revision conflict'); });
  let executed = false;
  const dependent = queue.enqueue(async () => { executed = true; });
  await expect(first).rejects.toThrow('revision conflict');
  await expect(dependent).rejects.toThrow('earlier edit failed');
  expect(executed).toBe(false);
  await expect(queue.enqueue(async () => 42)).resolves.toBe(42);
});
