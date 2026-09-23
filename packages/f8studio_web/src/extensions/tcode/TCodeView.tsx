import { useMemo } from 'react';

import type { JsonValue } from '../../api/contracts';

export function TCodeView({ payload }: { readonly payload: Readonly<Record<string, JsonValue>> }) {
  const line = typeof payload.line === 'string' ? payload.line : '';
  const values = useMemo(() => {
    const result = new Map<string, number>();
    for (const match of line.matchAll(/([A-Z]\d)(\d{1,5})/g)) result.set(match[1] ?? '', Math.max(0, Math.min(9999, Number(match[2]))));
    return result;
  }, [line]);
  return <div className="tcode-view"><code>{line || 'Waiting for TCode'}</code><div className="tcode-channels">{[...values.entries()].map(([channel, value]) => <label key={channel}><span>{channel}</span><meter min={0} max={9999} value={value} /><output>{value}</output></label>)}</div></div>;
}
