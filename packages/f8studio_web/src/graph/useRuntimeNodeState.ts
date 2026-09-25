import { useEffect, useMemo, useRef, useState } from 'react';

import { fetchRuntimeNodeState } from '../api/client';
import type { GraphNode, RuntimeStateField } from '../api/contracts';

export function useRuntimeNodeState(node: GraphNode, names: readonly string[]): Readonly<Record<string, RuntimeStateField>> {
  const [values, setValues] = useState<Readonly<Record<string, RuntimeStateField>>>({});
  const reportedError = useRef(false);
  const key = useMemo(() => names.join('\u0000'), [names]);
  useEffect(() => {
    const controller = new AbortController();
    const fields = key === '' ? [] : key.split('\u0000');
    setValues({});
    const load = async () => {
      if (fields.length === 0) {
        setValues({});
        return;
      }
      try {
        const state = await fetchRuntimeNodeState(node.serviceId, node.nodeId, fields, controller.signal);
        if (!controller.signal.aborted) {
          setValues(Object.fromEntries(state.fields.map((field) => [field.field, field])));
          reportedError.current = false;
        }
      } catch (reason: unknown) {
        if (controller.signal.aborted) return;
        setValues({});
        if (!reportedError.current) {
          reportedError.current = true;
          console.error(`Failed to read runtime state for ${node.nodeId}`, reason);
        }
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), 1500);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [node.nodeId, node.serviceId, key]);
  return values;
}
