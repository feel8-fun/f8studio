import { afterEach, expect, it, vi } from 'vitest';
import { deleteAsset, fetchProjects, setRuntimeState } from './client';

afterEach(() => vi.unstubAllGlobals());

it('reports validation locations from FastAPI errors', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
    detail: [{ loc: ['body', 'name'], msg: 'Field required' }],
  }), { status: 422 })));
  await expect(fetchProjects()).rejects.toThrow('body.name: Field required');
});

it('accepts empty successful deletes', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 204 })));
  await expect(deleteAsset('asset')).resolves.toBeUndefined();
});

it('rejects unsuccessful runtime actions even with HTTP 200', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
    success: false, result: null, errorMessage: 'Target is offline',
  }))));
  await expect(setRuntimeState('service', 'node', 'field', 1)).rejects.toThrow('Target is offline');
});
