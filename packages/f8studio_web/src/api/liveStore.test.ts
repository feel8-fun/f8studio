import { expect, test } from 'vitest';
import { LiveStore } from './liveStore';

test('keeps unrelated prefix snapshots stable and replaces state on reconnect snapshot', () => {
  const store = new LiveStore();
  store.apply({ type: 'live.snapshot', values: { 'state/a/x': 1, 'state/b/x': 2 } });
  const a = store.getPrefix('state/a/');
  const b = store.getPrefix('state/b/');
  store.apply({ type: 'live.patch', set: { 'state/a/x': 3 }, delete: [] });
  expect(store.getPrefix('state/a/')).not.toBe(a);
  expect(store.getPrefix('state/b/')).toBe(b);
  expect(store.getPrefix('state/a/').get('state/a/x')).toBe(3);
  store.apply({ type: 'live.snapshot', values: {} });
  expect(store.getPrefix('state/a/').size).toBe(0);
  expect(store.getPrefix('state/b/').size).toBe(0);
});
