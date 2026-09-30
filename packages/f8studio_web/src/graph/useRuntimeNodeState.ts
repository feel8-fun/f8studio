import { isJsonObject } from '../api/contracts';
import { useMemo } from 'react';
import type { GraphNode, RuntimeStateField } from '../api/contracts';
import { useLivePrefix } from '../api/liveStore';

export function useRuntimeNodeState(node: GraphNode, names: readonly string[]): Readonly<Record<string, RuntimeStateField>> {
  const prefix = `state/${node.serviceId}/${node.nodeId}/`;
  const values = useLivePrefix(prefix);
  const fields = names.join('\u0000');
  return useMemo(() => {
    const result: Record<string, RuntimeStateField> = {};
    for (const field of fields.split('\u0000')) {
      const value = values.get(prefix + field);
      if (isJsonObject(value) && value.field === field && typeof value.found === 'boolean') {
        result[field] = value as unknown as RuntimeStateField;
      }
    }
    return result;
  }, [prefix, fields, values]);
}
