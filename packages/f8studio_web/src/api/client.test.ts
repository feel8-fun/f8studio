import { afterEach, expect, it, vi } from 'vitest';
import { deleteAsset, fetchEnvironments, fetchExtensions, fetchProjects, importExtensionPackage,
  installExtension, setRuntimeState } from './client';

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

it.each(['native', 'bundled', 'workspace', 'pixi', 'shared'])('accepts %s extensions and environments', async (runtimeKind) => {
  const extension = { extensionId: 'tracker', name: 'Tracker', version: '1.0.0', description: 'Track objects',
    runtimeKind, environmentId: 'onnx-runtime', preinstalled: false, state: 'installed', detail: '',
    serviceClasses: ['thirdparty.tracker'] };
  vi.stubGlobal('fetch', vi.fn().mockImplementation(() => new Response(JSON.stringify([extension]))));
  await expect(fetchExtensions()).resolves.toEqual([extension]);
  await expect(importExtensionPackage('https://publisher.example/tracker.zip', 'a'.repeat(64))).resolves.toEqual([extension]);
  vi.stubGlobal('fetch', vi.fn().mockImplementation(() => new Response(JSON.stringify(extension))));
  await expect(installExtension('tracker')).resolves.toEqual(extension);
  const environment = { runtimeKind, environmentId: 'onnx-runtime', ready: true, extensionIds: ['tracker'] };
  vi.stubGlobal('fetch', vi.fn().mockImplementation(() => new Response(JSON.stringify([environment]))));
  await expect(fetchEnvironments()).resolves.toEqual([environment]);
});

it('rejects unknown extension runtime kinds', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify([{
    extensionId: 'tracker', name: 'Tracker', version: '1.0.0', description: '', runtimeKind: 'unknown',
    environmentId: null, preinstalled: false, state: 'available', detail: '', serviceClasses: [],
  }]))));
  await expect(fetchExtensions()).rejects.toThrow('Invalid extension status');
});
