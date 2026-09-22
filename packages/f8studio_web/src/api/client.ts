import {
  isAudioSessionAnswer,
  isHealthStatus,
  isMediaSample,
  isMediaSessionAnswer,
  isRtcConfigurationResponse,
  isGraphNode,
  isDeployJob,
  isProjectRecord,
  isStudioDocument,
  type CatalogSnapshot,
  type DeployJob,
  type GraphNode,
  type GraphOperation,
  type HealthStatus,
  type AudioSessionAnswer,
  type MediaSample,
  type MediaSessionAnswer,
  type RtcConfigurationResponse,
  type PatchResult,
  type ProjectRecord,
  type ProjectSummary,
  type RuntimeMonitor,
} from './contracts';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null = null,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function requestJson(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(path, init);
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    let message = `Request failed with HTTP ${response.status}`;
    let code: string | null = null;
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const detail = (body as { readonly detail: unknown }).detail;
      if (typeof detail === 'string') message = detail;
      if (typeof detail === 'object' && detail !== null) {
        const envelope = detail as Record<string, unknown>;
        if (typeof envelope.message === 'string') message = envelope.message;
        if (typeof envelope.code === 'string') code = envelope.code;
      }
    }
    throw new ApiError(message, response.status, code);
  }
  return body;
}

function jsonRequest(method: 'POST' | 'PUT', body: unknown, signal?: AbortSignal): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  };
}

export async function fetchProjects(signal?: AbortSignal): Promise<readonly ProjectSummary[]> {
  const body = await requestJson('/api/projects', { signal });
  if (!Array.isArray(body) || !body.every((item) => typeof item === 'object' && item !== null &&
    typeof (item as Record<string, unknown>).projectId === 'string' &&
    typeof (item as Record<string, unknown>).name === 'string')) {
    throw new Error('Project list does not match f8studio-api/1');
  }
  return body as unknown as readonly ProjectSummary[];
}

export async function createProject(name: string, signal?: AbortSignal): Promise<ProjectRecord> {
  const body = await requestJson('/api/projects', jsonRequest('POST', { name }, signal));
  if (!isProjectRecord(body)) throw new Error('Created project does not match f8studio-api/1');
  return body;
}

export async function fetchProject(projectId: string, signal?: AbortSignal): Promise<ProjectRecord> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}`, { signal });
  if (!isProjectRecord(body)) throw new Error('Project does not match f8studio-api/1');
  return body;
}

export async function fetchCatalog(signal?: AbortSignal): Promise<CatalogSnapshot> {
  const body = await requestJson('/api/catalog', { signal });
  if (typeof body !== 'object' || body === null) throw new Error('Catalog does not match f8studio-api/1');
  const item = body as Record<string, unknown>;
  if (!Array.isArray(item.services) || !Array.isArray(item.operators)) {
    throw new Error('Catalog does not match f8studio-api/1');
  }
  return body as unknown as CatalogSnapshot;
}

export interface CreateCatalogNodeInput {
  readonly kind: 'service' | 'operator';
  readonly nodeId: string;
  readonly serviceClass: string;
  readonly serviceId?: string;
  readonly operatorClass?: string;
}

export async function createCatalogNode(input: CreateCatalogNodeInput): Promise<GraphNode> {
  const body = await requestJson('/api/catalog/nodes', jsonRequest('POST', input));
  if (!isGraphNode(body)) throw new Error('Catalog node does not match f8studio-api/1');
  return body;
}

function isPatchResult(value: unknown): value is PatchResult {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return typeof item.requestId === 'string' && typeof item.graphChanged === 'boolean' &&
    typeof item.layoutChanged === 'boolean' && isStudioDocument(item.document);
}

export async function patchProject(
  projectId: string,
  document: { readonly graphRevision: number; readonly layoutRevision: number },
  operations: readonly GraphOperation[],
): Promise<PatchResult> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/patch`,
    jsonRequest('POST', {
      requestId: crypto.randomUUID(),
      expectedGraphRevision: document.graphRevision,
      expectedLayoutRevision: document.layoutRevision,
      operations,
    }),
  );
  if (!isPatchResult(body)) throw new Error('Patch result does not match f8studio-api/1');
  return body;
}

export async function changeHistory(
  projectId: string,
  action: 'undo' | 'redo',
  document: { readonly graphRevision: number; readonly layoutRevision: number },
): Promise<PatchResult> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/${action}`,
    jsonRequest('POST', {
      requestId: crypto.randomUUID(),
      expectedGraphRevision: document.graphRevision,
      expectedLayoutRevision: document.layoutRevision,
    }),
  );
  if (!isPatchResult(body)) throw new Error('History result does not match f8studio-api/1');
  return body;
}

export async function fetchLatestDeployment(projectId: string, signal?: AbortSignal): Promise<DeployJob | null> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}/deployments/latest`, { signal });
  if (body === null) return null;
  if (!isDeployJob(body)) throw new Error('Deployment does not match f8studio-api/1');
  return body;
}

export async function deployProject(projectId: string, graphRevision: number): Promise<DeployJob> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/deploy`,
    jsonRequest('POST', { requestId: crypto.randomUUID(), expectedGraphRevision: graphRevision }),
  );
  if (!isDeployJob(body)) throw new Error('Deployment does not match f8studio-api/1');
  return body;
}

export async function fetchDeployJob(jobId: string): Promise<DeployJob> {
  const body = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!isDeployJob(body)) throw new Error('Deployment does not match f8studio-api/1');
  return body;
}

export async function stopRuntimeService(serviceId: string): Promise<void> {
  await requestJson(`/api/runtime/services/${encodeURIComponent(serviceId)}/stop`, { method: 'POST' });
}

export async function fetchRuntimeMonitors(signal?: AbortSignal): Promise<readonly RuntimeMonitor[]> {
  const body = await requestJson('/api/runtime/monitors', { signal });
  if (!Array.isArray(body) || !body.every((value) => {
    if (typeof value !== 'object' || value === null) return false;
    const item = value as Record<string, unknown>;
    return typeof item.serviceId === 'string' && typeof item.nodeId === 'string' &&
      typeof item.tsMs === 'number' && typeof item.alive === 'boolean';
  })) throw new Error('Runtime monitors do not match f8studio-api/1');
  return body as unknown as readonly RuntimeMonitor[];
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  const response = await fetch('/api/health', { signal });
  if (!response.ok) {
    throw new Error(`Health request failed with HTTP ${response.status}`);
  }
  const body: unknown = await response.json();
  if (!isHealthStatus(body)) {
    throw new Error('Health response does not match f8studio-api/1');
  }
  return body;
}

export async function fetchRtcConfiguration(signal?: AbortSignal): Promise<RtcConfigurationResponse> {
  const response = await fetch('/api/media/rtc-configuration', { signal });
  if (!response.ok) throw new Error(`RTC configuration request failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isRtcConfigurationResponse(body)) throw new Error('RTC configuration does not match f8studio-api/1');
  return body;
}

export async function createMediaSession(
  source: string,
  quality: 'thumbnail' | 'main',
  description: RTCSessionDescriptionInit,
  overlay = false,
  signal?: AbortSignal,
): Promise<MediaSessionAnswer> {
  const response = await fetch('/api/media/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, quality, sdp: description.sdp, type: description.type, overlay }),
    signal,
  });
  if (!response.ok) throw new Error(`Media negotiation failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isMediaSessionAnswer(body)) throw new Error('Media answer does not match f8studio-api/1');
  return body;
}

export async function createAudioSession(
  source: string,
  description: RTCSessionDescriptionInit,
  signal?: AbortSignal,
): Promise<AudioSessionAnswer> {
  const response = await fetch('/api/audio/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, sdp: description.sdp, type: description.type }),
    signal,
  });
  if (!response.ok) throw new Error(`Audio negotiation failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isAudioSessionAnswer(body)) throw new Error('Audio answer does not match f8studio-api/1');
  return body;
}

export async function closeAudioSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/audio/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Audio session close failed with HTTP ${response.status}`);
  }
}

export async function closeMediaSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/media/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Media session close failed with HTTP ${response.status}`);
  }
}

export async function fetchMediaSample(source: string, x: number, y: number): Promise<MediaSample> {
  const query = new URLSearchParams({ source, x: String(x), y: String(y) });
  const response = await fetch(`/api/media/sample?${query.toString()}`);
  if (!response.ok) throw new Error(`Media sample failed with HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (!isMediaSample(body)) throw new Error('Media sample does not match f8studio-api/1');
  return body;
}
