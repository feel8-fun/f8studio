import type { JsonValue } from '../../api/contracts';

export function parseTCodeChannels(line: string): Readonly<Record<string, number>> {
  const channels: Record<string, number> = {};
  for (const match of line.matchAll(/(?:^|\s)([LRVA][0-9])(\d{1,4})(?=[IS\s]|$)/g)) {
    const channel = match[1];
    const value = match[2];
    if (channel !== undefined && value !== undefined) channels[channel] = Math.min(9999, Number(value));
  }
  return channels;
}

export function mergeTCodeChannels(previous: JsonValue | undefined, line: string): Readonly<Record<string, number>> {
  const channels: Record<string, number> = {};
  if (previous !== undefined && previous !== null && typeof previous === 'object' && !Array.isArray(previous)) {
    for (const [channel, value] of Object.entries(previous)) {
      if (typeof value === 'number' && Number.isFinite(value)) channels[channel] = value;
    }
  }
  return { ...channels, ...parseTCodeChannels(line) };
}
