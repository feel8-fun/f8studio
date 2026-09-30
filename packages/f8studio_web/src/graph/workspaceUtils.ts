

import type { ProjectRecord } from '../api/contracts';

export function newId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replaceAll('-', '')}`;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown graph operation error';
}

export function documentIsNewer(next: ProjectRecord['document'], current: ProjectRecord['document']): boolean {
  return next.graphRevision > current.graphRevision ||
    (next.graphRevision === current.graphRevision && next.layoutRevision > current.layoutRevision);
}
