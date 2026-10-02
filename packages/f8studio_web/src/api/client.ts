import type * as Wire from './contracts.gen';
import {
  isAudioSessionAnswer,
  isHealthStatus,
  isMediaSessionAnswer,
  isRtcConfigurationResponse,
  isGraphNode,
  isDeployJob,
  isProjectRecord,
  isStudioDocument,
  isStudioLogEvent,
  type CatalogSnapshot,
  type ExtensionStatus,
  type EnvironmentStatus,
  type DeployJob,
  type GraphNode,
  type GraphOperation,
  type HealthStatus,
  type AudioSessionAnswer,
  type MediaSessionAnswer,
  type RtcConfigurationResponse,
  type PatchResult,
  type ProjectRecord,
  type ProjectSummary,
  type StudioLogEvent,
  type PresentationCommand,
  type RuntimeMonitor,
  type RuntimeNodeState,
  type RuntimeActionResult,
  type AssetKind,
  type AssetRecord,
  type AssetSummary,
  type AssetVersion,
  type AgentProviderSummary,
  type AgentProviderSettings,
  type AgentConnectionProbe,
  type CreateAgentConnection,
  type AgentImage,
  type UpdateAgentProviderSettings,
  type AgentSession,
  type AgentSessionSummary,
  type EditorAnalysis,
  type EditorSession,
  type EditorLanguageResult,
  type HotkeyBinding,
  type JsonValue,
  type ProjectVersion,
  type RegisterHotkeyInput,
} from './contracts';

function isAgentSession(value: unknown): value is AgentSession {
  if (!isObject(value)) return false;
  return typeof value.sessionId === 'string' && typeof value.projectId === 'string' &&
    typeof value.title === 'string' && typeof value.status === 'string' &&
    Array.isArray(value.messages) && Array.isArray(value.toolCalls) && Array.isArray(value.artifacts);
}

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

async function requestJson(path: string, init?: RequestInit, ignoreNotFound = false): Promise<unknown> {
  const response = await fetch(path, init);
  if (ignoreNotFound && response.status === 404) return null;
  if (response.status === 204) return null;
  let body: unknown;
  try {
    body = await response.json();
  } catch (error: unknown) {
    throw new ApiError(`HTTP ${response.status}: invalid JSON response (${String(error)})`, response.status, 'invalid_response');
  }
  if (!response.ok) {
    let message = `Request failed with HTTP ${response.status}`;
    let code: string | null = null;
    const detail = isObject(body) && 'detail' in body ? body.detail : body;
    if (typeof detail === 'string') message = detail;
    if (Array.isArray(detail)) {
      message = detail.map((entry: unknown) => isObject(entry) && typeof entry.msg === 'string'
        ? `${Array.isArray(entry.loc) ? entry.loc.join('.') + ': ' : ''}${entry.msg}` : JSON.stringify(entry)).join('; ');
    } else if (isObject(detail)) {
      if (typeof detail.message === 'string') message = detail.message;
      if (typeof detail.code === 'string') code = detail.code;
    }
    throw new ApiError(message, response.status, code);
  }
  return body;
}

function jsonRequest<Route extends keyof Wire.ApiRequests>(route: Route, body: Wire.ApiRequests[Route], signal?: AbortSignal): RequestInit {
  return {
    method: route.slice(0, route.indexOf(' ')),
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

export async function fetchLogs(options: { readonly limit?: number; readonly beforeSequence?: number; readonly signal?: AbortSignal } = {}): Promise<readonly StudioLogEvent[]> {
  const query = new URLSearchParams();
  if (options.limit !== undefined) query.set('limit', String(options.limit));
  if (options.beforeSequence !== undefined) query.set('before_sequence', String(options.beforeSequence));
  const body = await requestJson(`/api/logs?${query.toString()}`, { signal: options.signal });
  if (!Array.isArray(body) || !body.every(isStudioLogEvent)) {
    throw new Error('Log history does not match f8studio-api/1');
  }
  return body;
}

export async function fetchAgentProviders(signal?: AbortSignal): Promise<readonly AgentProviderSummary[]> {
  const body = await requestJson('/api/agents/providers', { signal });
  if (!Array.isArray(body) || !body.every((value) => isObject(value) &&
    typeof value.providerId === 'string' && typeof value.displayName === 'string' &&
    typeof value.configured === 'boolean' && Array.isArray(value.models))) {
    throw new Error('Agent providers do not match f8studio-api/1');
  }
  return body as unknown as readonly AgentProviderSummary[];
}

function isProviderSettings(value: unknown): value is AgentProviderSettings {
  return isObject(value) && typeof value.providerId === 'string' && typeof value.displayName === 'string'
    && typeof value.model === 'string' && typeof value.endpoint === 'string'
    && typeof value.apiKeySet === 'boolean' && typeof value.requiresApiKey === 'boolean'
    && typeof value.configured === 'boolean' && (value.source === 'saved' || value.source === 'environment')
    && (value.kind === 'agent' || value.kind === 'decision')
    && typeof value.supportsImage === 'boolean'
    && Array.isArray(value.inputModalities) && value.inputModalities.every((item) => item === 'text' || item === 'image');
}

export async function fetchAgentProviderSettings(signal?: AbortSignal): Promise<readonly AgentProviderSettings[]> {
  const body = await requestJson('/api/agents/providers/settings', { signal });
  if (!Array.isArray(body) || !body.every(isProviderSettings)) throw new Error('Invalid agent provider settings');
  return body;
}

export async function saveAgentProviderSettings(providerId: string, input: UpdateAgentProviderSettings): Promise<AgentProviderSettings> {
  const body = await requestJson(`/api/agents/providers/${encodeURIComponent(providerId)}/settings`, jsonRequest('PUT /api/agents/providers/{provider_id}/settings', input));
  if (!isProviderSettings(body)) throw new Error('Invalid saved agent provider settings');
  return body;
}

export async function createAgentConnection(input: CreateAgentConnection): Promise<AgentProviderSettings> {
  const body = await requestJson('/api/agents/connections', jsonRequest('POST /api/agents/connections', input));
  if (!isProviderSettings(body)) throw new Error('Invalid created agent connection');
  return body;
}

export async function deleteAgentConnection(providerId: string): Promise<void> {
  await requestJson(`/api/agents/connections/${encodeURIComponent(providerId)}`, { method: 'DELETE' });
}

export async function probeAgentConnection(input: Wire.ProbeProviderRequestInput): Promise<AgentConnectionProbe> {
  const body = await requestJson('/api/agents/connections/probe', jsonRequest('POST /api/agents/connections/probe', input));
  if (!isObject(body) || typeof body.connected !== 'boolean' || !Array.isArray(body.models)
      || typeof body.detail !== 'string' || !['catalog', 'model', 'none'].includes(String(body.verified))) {
    throw new Error('Invalid connection probe result');
  }
  return body as unknown as AgentConnectionProbe;
}

export async function fetchAgentSessions(projectId: string, signal?: AbortSignal): Promise<readonly AgentSessionSummary[]> {
  const query = new URLSearchParams({ project_id: projectId });
  const body = await requestJson(`/api/agents/sessions?${query.toString()}`, { signal });
  if (!Array.isArray(body) || !body.every((value) => isObject(value) &&
    typeof value.sessionId === 'string' && typeof value.projectId === 'string' &&
    typeof value.status === 'string')) {
    throw new Error('Agent sessions do not match f8studio-api/1');
  }
  return body as unknown as readonly AgentSessionSummary[];
}

export async function fetchAgentSession(sessionId: string, signal?: AbortSignal): Promise<AgentSession> {
  const body = await requestJson(`/api/agents/sessions/${encodeURIComponent(sessionId)}`, { signal });
  if (!isAgentSession(body)) throw new Error('Agent session does not match f8studio-api/1');
  return body;
}

export async function createAgentSession(input: Wire.CreateAgentSessionRequestInput): Promise<AgentSession> {
  const body = await requestJson('/api/agents/sessions', jsonRequest('POST /api/agents/sessions', input));
  if (!isAgentSession(body)) throw new Error('Created agent session does not match f8studio-api/1');
  return body;
}

export async function renameAgentSession(sessionId: string, title: string): Promise<AgentSession> {
  const body = await requestJson(`/api/agents/sessions/${encodeURIComponent(sessionId)}`, jsonRequest('PUT /api/agents/sessions/{session_id}', { title }));
  if (!isAgentSession(body)) throw new Error('Renamed agent session does not match f8studio-api/1');
  return body;
}

export async function selectAgentModel(sessionId: string, providerId: string, modelId: string): Promise<AgentSession> {
  const body = await requestJson(`/api/agents/sessions/${encodeURIComponent(sessionId)}/model`,
    jsonRequest('PUT /api/agents/sessions/{session_id}/model', { providerId, modelId }));
  if (!isAgentSession(body)) throw new Error('Updated agent session does not match f8studio-api/1');
  return body;
}

export async function deleteAgentSession(sessionId: string): Promise<void> {
  await requestJson(`/api/agents/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
}

export async function startAgentRun(sessionId: string, prompt: string, images: readonly AgentImage[] = [], reasoningEffort?: Wire.StartAgentRunRequestInput['reasoningEffort']): Promise<AgentSession> {
  const body = await requestJson(
    `/api/agents/sessions/${encodeURIComponent(sessionId)}/runs`,
    jsonRequest('POST /api/agents/sessions/{session_id}/runs', { prompt, images, ...(reasoningEffort ? { reasoningEffort } : {}) }),
  );
  if (!isAgentSession(body)) throw new Error('Started agent session does not match f8studio-api/1');
  return body;
}

export async function resolveAgentApproval(
  sessionId: string,
  approvalId: string,
  argumentsHash: string,
  approved: boolean,
): Promise<AgentSession> {
  const body = await requestJson(
    `/api/agents/sessions/${encodeURIComponent(sessionId)}/approvals/${encodeURIComponent(approvalId)}`,
    jsonRequest('POST /api/agents/sessions/{session_id}/approvals/{approval_id}', { approved, argumentsHash }),
  );
  if (!isAgentSession(body)) throw new Error('Resolved agent session does not match f8studio-api/1');
  return body;
}

export async function cancelAgentRun(sessionId: string): Promise<AgentSession> {
  const body = await requestJson(
    `/api/agents/sessions/${encodeURIComponent(sessionId)}/runs/current`,
    { method: 'DELETE' },
  );
  if (!isAgentSession(body)) throw new Error('Cancelled agent session does not match f8studio-api/1');
  return body;
}

export async function createProject(name: string, signal?: AbortSignal): Promise<ProjectRecord> {
  const body = await requestJson('/api/projects', jsonRequest('POST /api/projects', { name }, signal));
  if (!isProjectRecord(body)) throw new Error('Created project does not match f8studio-api/1');
  return body;
}

export async function deleteProject(projectId: string): Promise<void> {
  await requestJson(`/api/projects/${encodeURIComponent(projectId)}`, { method: 'DELETE' });
}

export async function fetchProject(projectId: string, signal?: AbortSignal): Promise<ProjectRecord> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}`, { signal });
  if (!isProjectRecord(body)) throw new Error('Project does not match f8studio-api/1');
  return body;
}

export async function exportProjectGraph(projectId: string): Promise<string> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}/graph/export`);
  if (!isObject(body) || body.format !== 'f8graph' || body.formatVersion !== 3) {
    throw new Error('Graph export does not match f8graph/3');
  }
  return `${JSON.stringify(body, null, 2)}\n`;
}

export async function importProjectGraph(projectId: string, content: string, expected: Pick<ProjectRecord['document'], 'graphRevision' | 'layoutRevision'>): Promise<ProjectRecord> {
  const query = new URLSearchParams({ expected_graph_revision: String(expected.graphRevision), expected_layout_revision: String(expected.layoutRevision) });
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}/graph/import?${query}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: content,
  });
  if (!isProjectRecord(body)) throw new Error('Imported project does not match f8studio-api/1');
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

export async function refreshCatalog(): Promise<CatalogSnapshot> {
  const body = await requestJson('/api/catalog/refresh', { method: 'POST' });
  if (typeof body !== 'object' || body === null) throw new Error('Catalog does not match f8studio-api/1');
  const item = body as Record<string, unknown>;
  if (!Array.isArray(item.services) || !Array.isArray(item.operators)) {
    throw new Error('Catalog does not match f8studio-api/1');
  }
  return body as unknown as CatalogSnapshot;
}

function isRuntimeKind(value: unknown): value is ExtensionStatus['runtimeKind'] {
  return typeof value === 'string' && ['native', 'bundled', 'workspace', 'pixi', 'shared'].includes(value);
}

function isExtensionStatus(value: unknown): value is ExtensionStatus {
  return isObject(value) && typeof value.extensionId === 'string' && typeof value.name === 'string' &&
    typeof value.version === 'string' && typeof value.description === 'string' &&
    isRuntimeKind(value.runtimeKind) &&
    (value.environmentId === null || typeof value.environmentId === 'string') && typeof value.preinstalled === 'boolean' &&
    ['unavailable', 'available', 'installing', 'installed', 'disabled', 'failed'].includes(String(value.state)) &&
    typeof value.detail === 'string' && Array.isArray(value.serviceClasses) &&
    value.serviceClasses.every((item) => typeof item === 'string');
}

export async function fetchExtensions(signal?: AbortSignal): Promise<readonly ExtensionStatus[]> {
  const body = await requestJson('/api/extensions', { signal });
  if (!Array.isArray(body) || !body.every(isExtensionStatus)) throw new Error('Invalid extension status');
  return body;
}

export async function importExtensionPackage(url: string, sha256: string): Promise<readonly ExtensionStatus[]> {
  const body = await requestJson('/api/extensions/import', jsonRequest('POST /api/extensions/import', { url, sha256 }));
  if (!Array.isArray(body) || !body.every(isExtensionStatus)) throw new Error('Invalid extension catalog');
  return body;
}

export async function fetchEnvironments(signal?: AbortSignal): Promise<readonly EnvironmentStatus[]> {
  const body = await requestJson('/api/environments', { signal });
  if (!Array.isArray(body) || !body.every((item: unknown) => isObject(item) &&
    typeof item.environmentId === 'string' && typeof item.ready === 'boolean' &&
    isRuntimeKind(item.runtimeKind) &&
    Array.isArray(item.extensionIds) && item.extensionIds.every((id: unknown) => typeof id === 'string'))) {
    throw new Error('Invalid environment status');
  }
  return body as readonly EnvironmentStatus[];
}

export async function installExtension(extensionId: string): Promise<ExtensionStatus> {
  const body = await requestJson(`/api/extensions/${encodeURIComponent(extensionId)}/install`, { method: 'POST' });
  if (!isExtensionStatus(body)) throw new Error('Invalid extension status');
  return body;
}

export async function cancelExtensionInstall(extensionId: string): Promise<ExtensionStatus> {
  const body = await requestJson(`/api/extensions/${encodeURIComponent(extensionId)}/cancel`, { method: 'POST' });
  if (!isExtensionStatus(body)) throw new Error('Invalid extension status');
  return body;
}

export async function setExtensionEnabled(extensionId: string, enabled: boolean): Promise<ExtensionStatus> {
  const body = await requestJson(`/api/extensions/${encodeURIComponent(extensionId)}/enabled`,
    jsonRequest('PUT /api/extensions/{extension_id}/enabled', { enabled }));
  if (!isExtensionStatus(body)) throw new Error('Invalid extension status');
  return body;
}

export async function uninstallExtension(extensionId: string): Promise<ExtensionStatus> {
  const body = await requestJson(`/api/extensions/${encodeURIComponent(extensionId)}`, { method: 'DELETE' });
  if (!isExtensionStatus(body)) throw new Error('Invalid extension status');
  return body;
}

export type CreateCatalogNodeInput = Wire.CreateCatalogNodeRequestInput;

export async function createCatalogNode(input: CreateCatalogNodeInput): Promise<GraphNode> {
  const body = await requestJson('/api/catalog/nodes', jsonRequest('POST /api/catalog/nodes', input));
  if (!isGraphNode(body)) throw new Error('Catalog node does not match f8studio-api/1');
  return body;
}

function isPatchResult(value: unknown): value is PatchResult {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return typeof item.requestId === 'string' && typeof item.graphChanged === 'boolean' &&
    typeof item.layoutChanged === 'boolean' && isStudioDocument(item.document) &&
    Array.isArray(item.runtimeErrors) && item.runtimeErrors.every((error) => typeof error === 'string');
}

export async function patchProject(
  projectId: string,
  document: Pick<Wire.StudioDocument, 'graphRevision' | 'layoutRevision'>,
  operations: readonly GraphOperation[],
): Promise<PatchResult> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/patch`,
    jsonRequest('POST /api/projects/{project_id}/patch', {
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
  document: Pick<Wire.StudioDocument, 'graphRevision' | 'layoutRevision'>,
): Promise<PatchResult> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/${action}`,
    jsonRequest(`POST /api/projects/{project_id}/${action}`, {
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
    jsonRequest('POST /api/projects/{project_id}/deploy', { requestId: crypto.randomUUID(), expectedGraphRevision: graphRevision }),
  );
  if (!isDeployJob(body)) throw new Error('Deployment does not match f8studio-api/1');
  return body;
}

export async function restartProjectService(projectId: string, serviceId: string): Promise<DeployJob> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/services/${encodeURIComponent(serviceId)}/restart`,
    { method: 'POST' },
  );
  if (!isDeployJob(body)) throw new Error('Restart deployment does not match f8studio-api/1');
  return body;
}

export async function fetchDeployJob(jobId: string): Promise<DeployJob> {
  const body = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!isDeployJob(body)) throw new Error('Deployment does not match f8studio-api/1');
  return body;
}

export async function stopProject(projectId: string): Promise<void> {
  await requestJson(`/api/projects/${encodeURIComponent(projectId)}/stop`, { method: 'POST' });
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

export async function fetchRuntimeNodeState(
  serviceId: string,
  nodeId: string,
  fields: readonly string[],
  signal?: AbortSignal,
): Promise<RuntimeNodeState> {
  const body = await requestJson(
    `/api/runtime/services/${encodeURIComponent(serviceId)}/nodes/${encodeURIComponent(nodeId)}/state:read`,
    jsonRequest('POST /api/runtime/services/{service_id}/nodes/{node_id}/state:read', { fields }, signal),
  );
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    throw new Error('Runtime node state does not match f8studio-api/1');
  }
  const state = body as Record<string, unknown>;
  if (state.serviceId !== serviceId || state.nodeId !== nodeId || !Array.isArray(state.fields) ||
      !state.fields.every((entry) => typeof entry === 'object' && entry !== null &&
        typeof (entry as Record<string, unknown>).field === 'string' &&
        typeof (entry as Record<string, unknown>).found === 'boolean')) {
    throw new Error('Runtime node state does not match f8studio-api/1');
  }
  return body as unknown as RuntimeNodeState;
}

export async function fetchPresentationSnapshot(signal?: AbortSignal): Promise<readonly PresentationCommand[]> {
  const body = await requestJson('/api/presentation', { signal });
  if (!Array.isArray(body) || !body.every((value) => {
    if (typeof value !== 'object' || value === null) return false;
    const command = value as Record<string, unknown>;
    return typeof command.nodeId === 'string' && typeof command.command === 'string' &&
      typeof command.payload === 'object' && command.payload !== null && !Array.isArray(command.payload);
  })) throw new Error('Presentation snapshot does not match f8studio-api/1');
  return body as unknown as readonly PresentationCommand[];
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  const body = await requestJson('/api/health', { signal });
  if (!isHealthStatus(body)) {
    throw new Error('Health response does not match f8studio-api/1');
  }
  return body;
}

export async function fetchRtcConfiguration(signal?: AbortSignal): Promise<RtcConfigurationResponse> {
  const body = await requestJson('/api/media/rtc-configuration', { signal });
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
  if (description.type !== 'offer' || typeof description.sdp !== 'string') {
    throw new Error('WebRTC session creation requires an SDP offer');
  }
  const body = await requestJson('/api/media/sessions', jsonRequest('POST /api/media/sessions', { source, quality, sdp: description.sdp, type: description.type, overlay }, signal));
  if (!isMediaSessionAnswer(body)) throw new Error('Media answer does not match f8studio-api/1');
  return body;
}

export async function createAudioSession(
  source: string,
  description: RTCSessionDescriptionInit,
  signal?: AbortSignal,
): Promise<AudioSessionAnswer> {
  if (description.type !== 'offer' || typeof description.sdp !== 'string') {
    throw new Error('WebRTC session creation requires an SDP offer');
  }
  const body = await requestJson('/api/audio/sessions', jsonRequest('POST /api/audio/sessions', { source, sdp: description.sdp, type: description.type }, signal));
  if (!isAudioSessionAnswer(body)) throw new Error('Audio answer does not match f8studio-api/1');
  return body;
}

export async function closeAudioSession(sessionId: string, keepalive = false): Promise<void> {
  await requestJson(`/api/audio/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE', keepalive }, true);
}

export async function closeMediaSession(sessionId: string, keepalive = false): Promise<void> {
  await requestJson(`/api/media/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE', keepalive }, true);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export async function fetchAssets(kind?: AssetKind, signal?: AbortSignal): Promise<readonly AssetSummary[]> {
  const query = kind === undefined ? '' : `?kind=${encodeURIComponent(kind)}`;
  const body = await requestJson(`/api/assets${query}`, { signal });
  if (!Array.isArray(body)) throw new Error('Asset list does not match f8studio-api/1');
  return body as readonly AssetSummary[];
}

export async function fetchAsset(assetId: string, signal?: AbortSignal): Promise<AssetRecord> {
  const body = await requestJson(`/api/assets/${encodeURIComponent(assetId)}`, { signal });
  if (!isObject(body) || typeof body.assetId !== 'string') throw new Error('Asset does not match f8studio-api/1');
  return body as unknown as AssetRecord;
}

export async function createAsset(input: Wire.CreateAssetRequestInput): Promise<AssetRecord> {
  const body = await requestJson('/api/assets', jsonRequest('POST /api/assets', input));
  if (!isObject(body) || typeof body.assetId !== 'string') throw new Error('Created asset does not match f8studio-api/1');
  return body as unknown as AssetRecord;
}

export async function updateAsset(assetId: string, input: Wire.UpdateAssetRequestInput): Promise<AssetRecord> {
  const body = await requestJson(`/api/assets/${encodeURIComponent(assetId)}`, jsonRequest('PUT /api/assets/{asset_id}', input));
  if (!isObject(body) || typeof body.assetId !== 'string') throw new Error('Updated asset does not match f8studio-api/1');
  return body as unknown as AssetRecord;
}

export async function deleteAsset(assetId: string): Promise<void> {
  await requestJson(`/api/assets/${encodeURIComponent(assetId)}`, { method: 'DELETE' });
}

export async function fetchAssetVersions(assetId: string): Promise<readonly AssetVersion[]> {
  const body = await requestJson(`/api/assets/${encodeURIComponent(assetId)}/versions`);
  if (!Array.isArray(body)) throw new Error('Asset versions do not match f8studio-api/1');
  return body as readonly AssetVersion[];
}

export async function createProjectVersion(projectId: string, name: string): Promise<ProjectVersion> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}/versions`, jsonRequest('POST /api/projects/{project_id}/versions', { name }));
  if (!isObject(body) || typeof body.versionId !== 'string') throw new Error('Project version does not match f8studio-api/1');
  return body as unknown as ProjectVersion;
}

export async function fetchProjectVersions(projectId: string): Promise<readonly ProjectVersion[]> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}/versions`);
  if (!Array.isArray(body)) throw new Error('Project versions do not match f8studio-api/1');
  return body as readonly ProjectVersion[];
}

export async function restoreProjectVersion(projectId: string, versionId: string): Promise<ProjectRecord> {
  const body = await requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}/restore`,
    { method: 'POST' },
  );
  if (!isProjectRecord(body)) throw new Error('Restored project does not match f8studio-api/1');
  return body;
}

export async function createEditorSession(
  language: 'python' | 'json',
  text: string,
  filename: string,
  target?: Pick<Wire.CreateEditorSessionRequestInput, 'projectId' | 'nodeId' | 'fieldName'>,
): Promise<EditorSession> {
  const body = await requestJson('/api/editor/sessions', jsonRequest('POST /api/editor/sessions', { language, text, filename, ...target }));
  if (!isObject(body) || typeof body.sessionId !== 'string') throw new Error('Editor session does not match f8studio-api/1');
  return body as unknown as EditorSession;
}

export async function updateEditorSession(sessionId: string, version: number, text: string): Promise<EditorSession> {
  const body = await requestJson(
    `/api/editor/sessions/${encodeURIComponent(sessionId)}`,
    jsonRequest('PUT /api/editor/sessions/{session_id}', { version, text }),
  );
  if (!isObject(body) || typeof body.sessionId !== 'string') throw new Error('Editor session does not match f8studio-api/1');
  return body as unknown as EditorSession;
}

export async function analyzeEditorSession(sessionId: string): Promise<EditorAnalysis> {
  const body = await requestJson(`/api/editor/sessions/${encodeURIComponent(sessionId)}/analyze`, { method: 'POST' });
  if (!isObject(body) || !Array.isArray(body.diagnostics)) throw new Error('Editor diagnostics do not match f8studio-api/1');
  return body as unknown as EditorAnalysis;
}

async function requestEditorLanguage(
  sessionId: string,
  operation: 'completion' | 'hover' | 'signature-help',
  line: number,
  column: number,
): Promise<EditorLanguageResult> {
  const body = await requestJson(
    `/api/editor/sessions/${encodeURIComponent(sessionId)}/${operation}`,
    jsonRequest(`POST /api/editor/sessions/{session_id}/${operation}`, { line, column }),
  );
  if (!isObject(body) || typeof body.sessionId !== 'string' || !('result' in body)) {
    throw new Error(`Editor ${operation} does not match f8studio-api/1`);
  }
  return body as unknown as EditorLanguageResult;
}

export async function requestEditorCompletion(sessionId: string, line: number, column: number): Promise<EditorLanguageResult> {
  return requestEditorLanguage(sessionId, 'completion', line, column);
}

export async function requestEditorHover(sessionId: string, line: number, column: number): Promise<EditorLanguageResult> {
  return requestEditorLanguage(sessionId, 'hover', line, column);
}

export async function requestEditorSignatureHelp(sessionId: string, line: number, column: number): Promise<EditorLanguageResult> {
  return requestEditorLanguage(sessionId, 'signature-help', line, column);
}

export async function closeEditorSession(sessionId: string): Promise<void> {
  await requestJson(`/api/editor/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }, true);
}

export async function fetchHotkeys(projectId?: string): Promise<readonly HotkeyBinding[]> {
  const query = projectId === undefined ? '' : `?project_id=${encodeURIComponent(projectId)}`;
  const body = await requestJson(`/api/local/hotkeys${query}`);
  if (!Array.isArray(body)) throw new Error('Hotkeys do not match f8studio-api/1');
  return body as readonly HotkeyBinding[];
}

export async function registerHotkey(input: RegisterHotkeyInput): Promise<HotkeyBinding> {
  const body = await requestJson('/api/local/hotkeys', jsonRequest('POST /api/local/hotkeys', input));
  if (!isObject(body) || typeof body.bindingId !== 'string') throw new Error('Hotkey does not match f8studio-api/1');
  return body as unknown as HotkeyBinding;
}

export async function unregisterHotkey(bindingId: string): Promise<void> {
  await requestJson(`/api/local/hotkeys/${encodeURIComponent(bindingId)}`, { method: 'DELETE' });
}

export async function invokeRuntimeCommand(serviceId: string, call: string, params: Readonly<Record<string, JsonValue>>): Promise<RuntimeActionResult> {
  return requireRuntimeAction(await requestJson(
    `/api/runtime/services/${encodeURIComponent(serviceId)}/commands`,
    jsonRequest('POST /api/runtime/services/{service_id}/commands', { call, params }),
  ));
}

export async function setRuntimeState(serviceId: string, nodeId: string, field: string, value: JsonValue): Promise<RuntimeActionResult> {
  return requireRuntimeAction(await requestJson(
    `/api/runtime/services/${encodeURIComponent(serviceId)}/state`,
    jsonRequest('POST /api/runtime/services/{service_id}/state', { nodeId, field, value }),
  ));
}

function requireRuntimeAction(value: unknown): RuntimeActionResult {
  if (!isObject(value) || typeof value.success !== 'boolean' || typeof value.errorMessage !== 'string'
      || !('result' in value)) throw new ApiError('Invalid runtime action response', 502, 'invalid_response');
  if (!value.success) throw new ApiError(value.errorMessage || 'Runtime action failed', 422, 'runtime_action_failed');
  return value as unknown as RuntimeActionResult;
}

function isToolRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isToolView(value: unknown): value is Wire.ToolView {
  if (!isToolRecord(value)) return false;
  return typeof value.extensionId === 'string' && typeof value.toolId === 'string' &&
    typeof value.name === 'string' && typeof value.description === 'string' &&
    typeof value.requiresConfirmation === 'boolean' && typeof value.allowConcurrent === 'boolean' && Array.isArray(value.fields) &&
    value.fields.every((field: unknown) => isToolRecord(field) && typeof field.name === 'string' &&
      typeof field.label === 'string' && typeof field.required === 'boolean' &&
      (field.kind === 'string' || field.kind === 'integer' || field.kind === 'number' || field.kind === 'boolean') &&
      Array.isArray(field.choices) && field.choices.every((choice: unknown) => typeof choice === 'string') && 'default' in field);
}

function isToolJob(value: unknown): value is Wire.ToolJob {
  if (!isToolRecord(value)) return false;
  const result = value.result;
  return typeof value.jobId === 'string' && typeof value.extensionId === 'string' &&
    typeof value.extensionVersion === 'string' && typeof value.toolId === 'string' && isToolRecord(value.arguments) &&
    typeof value.createdAt === 'string' && typeof value.updatedAt === 'string' &&
    typeof value.error === 'string' && typeof value.log === 'string' &&
    (value.status === 'queued' || value.status === 'running' || value.status === 'succeeded' || value.status === 'failed' || value.status === 'cancelled') &&
    (result === null || (isToolRecord(result) && result.schemaVersion === 'f8toolResult/1' &&
      typeof result.success === 'boolean' && typeof result.message === 'string' && 'data' in result));
}

export async function fetchExtensionTools(signal?: AbortSignal): Promise<readonly Wire.ToolView[]> {
  const body = await requestJson('/api/extension-tools', { signal });
  if (!Array.isArray(body) || !body.every(isToolView)) throw new Error('Invalid extension tools');
  return body;
}
export async function fetchToolJobs(signal?: AbortSignal): Promise<readonly Wire.ToolJob[]> {
  const body = await requestJson('/api/tool-jobs', { signal });
  if (!Array.isArray(body) || !body.every(isToolJob)) throw new Error('Invalid tool jobs');
  return body;
}
export async function runExtensionTool(extensionId: string, toolId: string, arguments_: Readonly<Record<string, Wire.JsonValue>>, confirm: boolean): Promise<Wire.ToolJob> {
  const body = await requestJson(`/api/extension-tools/${encodeURIComponent(extensionId)}/${encodeURIComponent(toolId)}/run`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ arguments: arguments_, confirm }),
  });
  if (!isToolJob(body)) throw new Error('Invalid tool job');
  return body;
}
export async function cancelToolJob(jobId: string): Promise<Wire.ToolJob> {
  const body = await requestJson(`/api/tool-jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  if (!isToolJob(body)) throw new Error('Invalid tool job');
  return body;
}
