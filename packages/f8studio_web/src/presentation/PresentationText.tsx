import type { JsonValue } from '../api/contracts';

export function PresentationText({ payload }: { readonly payload: Readonly<Record<string, JsonValue>> }) {
  return <pre>{JSON.stringify(payload.value, null, 2)}</pre>;
}
