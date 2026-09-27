import { useCallback, useSyncExternalStore } from 'react';

export type TCodeViewMode = '3d' | 'bars';

const listeners = new Map<string, Set<() => void>>();

function storageKey(nodeId: string): string {
  return `f8studio.tcodeView.${nodeId}`;
}

function getViewMode(nodeId: string): TCodeViewMode {
  return localStorage.getItem(storageKey(nodeId)) === 'bars' ? 'bars' : '3d';
}

function subscribeViewMode(nodeId: string, listener: () => void): () => void {
  const subscribed = listeners.get(nodeId) ?? new Set<() => void>();
  subscribed.add(listener);
  listeners.set(nodeId, subscribed);
  const onStorage = (event: StorageEvent) => {
    if (event.key === storageKey(nodeId)) listener();
  };
  window.addEventListener('storage', onStorage);
  return () => {
    subscribed.delete(listener);
    if (subscribed.size === 0) listeners.delete(nodeId);
    window.removeEventListener('storage', onStorage);
  };
}

export function setTCodeViewMode(nodeId: string, mode: TCodeViewMode): void {
  localStorage.setItem(storageKey(nodeId), mode);
  for (const listener of listeners.get(nodeId) ?? []) listener();
}

export function useTCodeViewMode(nodeId: string): TCodeViewMode {
  const subscribe = useCallback((listener: () => void) => subscribeViewMode(nodeId, listener), [nodeId]);
  const getSnapshot = useCallback(() => getViewMode(nodeId), [nodeId]);
  return useSyncExternalStore(subscribe, getSnapshot, () => '3d');
}
