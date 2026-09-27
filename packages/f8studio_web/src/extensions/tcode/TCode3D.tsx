import { useEffect, useRef, useState } from 'react';
import { OSREmulator, type OSRModel } from 'osr-emu';

import type { JsonValue } from '../../api/contracts';

export function TCode3D({ payload, line, model, resetVersion }: {
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly line: string;
  readonly model: OSRModel;
  readonly resetVersion: number;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const emulatorRef = useRef<OSREmulator | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (host === null) return;
    try {
      emulatorRef.current = new OSREmulator(host, { model });
      setError(null);
    } catch (reason) {
      console.error(`Failed to start ${model} TCode visualizer`, reason);
      setError(`Unable to display ${model} model`);
    }
    return () => {
      emulatorRef.current?.destroy();
      emulatorRef.current = null;
    };
  }, [model, resetVersion]);

  useEffect(() => {
    if (line === '' || emulatorRef.current === null) return;
    try {
      emulatorRef.current.write(line.endsWith('\n') ? line : `${line}\n`);
    } catch (reason) {
      console.error('Failed to update TCode visualizer', reason);
      setError('Unable to apply TCode command');
    }
  }, [payload, line, model, resetVersion]);

  return <>
    <div ref={hostRef} className="tcode-stage" aria-label={`${model} 3D TCode visualizer`} />
    {error !== null && <span className="tcode-error" role="status">{error}</span>}
  </>;
}
