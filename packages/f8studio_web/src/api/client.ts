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
  isStudioLogEvent,
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
  type StudioLogEvent,
  type PresentationCommand,
  type RuntimeMonitor,
  type RuntimeNodeState,
  type AssetKind,
  type AssetRecord,
  type AssetSummary,
  type AssetVersion,
  type AgentProviderSummary,
  type AgentSession,
  type AgentSessionSummary,
  type EditorAnalysis,
  type EditorSession,
  type EditorLanguageResult,
  type HotkeyBinding,
  type JsonValue,
  type LocalCapability,
  type ProjectVersion,
  type RegisterHotkeyInput,
  type SerialPortInfo,
  type SkeletonUdpVerification,
  type UnityInstallPlan,
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

export async function fetchLogs(signal?: AbortSignal): Promise<readonly StudioLogEvent[]> {
  const body = await requestJson('/api/logs', { signal });
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

export async function createAgentSession(input: {
  readonly projectId: string;
  readonly title: string;
  readonly providerId: string;
  readonly modelId: string;
}): Promise<AgentSession> {
  const body = await requestJson('/api/agents/sessions', jsonRequest('POST', input));
  if (!isAgentSession(body)) throw new Error('Created agent session does not match f8studio-api/1');
  return body;
}

export async function startAgentRun(sessionId: string, prompt: string): Promise<AgentSession> {
  const body = await requestJson(
    `/api/agents/sessions/${encodeURIComponent(sessionId)}/runs`,
    jsonRequest('POST', { prompt }),
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
    jsonRequest('POST', { approved, argumentsHash }),
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

export async function fetchRuntimeNodeState(
  serviceId: string,
  nodeId: string,
  fields: readonly string[],
  signal?: AbortSignal,
): Promise<RuntimeNodeState> {
  const body = await requestJson(
    `/api/runtime/services/${encodeURIComponent(serviceId)}/nodes/${encodeURIComponent(nodeId)}/state:read`,
    jsonRequest('POST', { fields }, signal),
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

export async function closeMediaSession(sessionId: string, keepalive = false): Promise<void> {
  const response = await fetch(`/api/media/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE', keepalive });
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

export async function createAsset(input: {
  readonly kind: AssetKind;
  readonly name: string;
  readonly description?: string;
  readonly tags?: readonly string[];
  readonly content: JsonValue;
}): Promise<AssetRecord> {
  const body = await requestJson('/api/assets', jsonRequest('POST', input));
  if (!isObject(body) || typeof body.assetId !== 'string') throw new Error('Created asset does not match f8studio-api/1');
  return body as unknown as AssetRecord;
}

export async function updateAsset(assetId: string, input: {
  readonly name: string;
  readonly description?: string;
  readonly tags?: readonly string[];
  readonly content: JsonValue;
}): Promise<AssetRecord> {
  const body = await requestJson(`/api/assets/${encodeURIComponent(assetId)}`, jsonRequest('PUT', input));
  if (!isObject(body) || typeof body.assetId !== 'string') throw new Error('Updated asset does not match f8studio-api/1');
  return body as unknown as AssetRecord;
}

export async function deleteAsset(assetId: string): Promise<void> {
  const response = await fetch(`/api/assets/${encodeURIComponent(assetId)}`, { method: 'DELETE' });
  if (!response.ok) throw new ApiError(`Delete failed with HTTP ${response.status}`, response.status);
}

export async function fetchAssetVersions(assetId: string): Promise<readonly AssetVersion[]> {
  const body = await requestJson(`/api/assets/${encodeURIComponent(assetId)}/versions`);
  if (!Array.isArray(body)) throw new Error('Asset versions do not match f8studio-api/1');
  return body as readonly AssetVersion[];
}

export async function createProjectVersion(projectId: string, name: string): Promise<ProjectVersion> {
  const body = await requestJson(`/api/projects/${encodeURIComponent(projectId)}/versions`, jsonRequest('POST', { name }));
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
): Promise<EditorSession> {
  const body = await requestJson('/api/editor/sessions', jsonRequest('POST', { language, text, filename }));
  if (!isObject(body) || typeof body.sessionId !== 'string') throw new Error('Editor session does not match f8studio-api/1');
  return body as unknown as EditorSession;
}

export async function updateEditorSession(sessionId: string, version: number, text: string): Promise<EditorSession> {
  const body = await requestJson(
    `/api/editor/sessions/${encodeURIComponent(sessionId)}`,
    jsonRequest('PUT', { version, text }),
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
  operation: 'completion' | 'hover',
  line: number,
  column: number,
): Promise<EditorLanguageResult> {
  const body = await requestJson(
    `/api/editor/sessions/${encodeURIComponent(sessionId)}/${operation}`,
    jsonRequest('POST', { line, column }),
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

export async function closeEditorSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/editor/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) throw new ApiError(`Editor close failed with HTTP ${response.status}`, response.status);
}

export async function fetchLocalCapabilities(signal?: AbortSignal): Promise<readonly LocalCapability[]> {
  const body = await requestJson('/api/local/capabilities', { signal });
  if (!Array.isArray(body)) throw new Error('Local capabilities do not match f8studio-api/1');
  return body as readonly LocalCapability[];
}

export async function fetchSerialPorts(): Promise<readonly SerialPortInfo[]> {
  const body = await requestJson('/api/local/serial-ports');
  if (!Array.isArray(body)) throw new Error('Serial ports do not match f8studio-api/1');
  return body as readonly SerialPortInfo[];
}

export async function detectModdingTarget(targetPath: string): Promise<Readonly<Record<string, JsonValue>>> {
  const body = await requestJson('/api/local/modding/detect', jsonRequest('POST', { targetPath }));
  if (!isObject(body)) throw new Error('Modding detection does not match f8studio-api/1');
  return body as Readonly<Record<string, JsonValue>>;
}

export async function previewUnityInstall(targetPath: string): Promise<UnityInstallPlan> {
  const body = await requestJson('/api/local/modding/unity/preview', jsonRequest('POST', { targetPath, offline: true }));
  if (!isObject(body) || typeof body.planId !== 'string') throw new Error('Unity plan does not match f8studio-api/1');
  return body as unknown as UnityInstallPlan;
}

export async function applyUnityInstall(planId: string, confirm: boolean): Promise<JsonValue> {
  return await requestJson('/api/local/modding/unity/apply', jsonRequest('POST', { planId, confirm })) as JsonValue;
}

export async function verifySkeletonUdp(port: number): Promise<SkeletonUdpVerification> {
  const body = await requestJson('/api/local/modding/verify-udp', jsonRequest('POST', { port }));
  if (!isObject(body) || typeof body.verified !== 'boolean') throw new Error('UDP verification does not match f8studio-api/1');
  return body as unknown as SkeletonUdpVerification;
}

export async function fetchHotkeys(projectId?: string): Promise<readonly HotkeyBinding[]> {
  const query = projectId === undefined ? '' : `?project_id=${encodeURIComponent(projectId)}`;
  const body = await requestJson(`/api/local/hotkeys${query}`);
  if (!Array.isArray(body)) throw new Error('Hotkeys do not match f8studio-api/1');
  return body as readonly HotkeyBinding[];
}

export async function registerHotkey(input: RegisterHotkeyInput): Promise<HotkeyBinding> {
  const body = await requestJson('/api/local/hotkeys', jsonRequest('POST', input));
  if (!isObject(body) || typeof body.bindingId !== 'string') throw new Error('Hotkey does not match f8studio-api/1');
  return body as unknown as HotkeyBinding;
}

export async function unregisterHotkey(bindingId: string): Promise<void> {
  const response = await fetch(`/api/local/hotkeys/${encodeURIComponent(bindingId)}`, { method: 'DELETE' });
  if (!response.ok) throw new ApiError(`Hotkey delete failed with HTTP ${response.status}`, response.status);
}

export async function invokeRuntimeCommand(serviceId: string, call: string, params: Readonly<Record<string, JsonValue>>): Promise<JsonValue> {
  return await requestJson(
    `/api/runtime/services/${encodeURIComponent(serviceId)}/commands`,
    jsonRequest('POST', { call, params }),
  ) as JsonValue;
}

export async function setRuntimeState(serviceId: string, nodeId: string, field: string, value: JsonValue): Promise<JsonValue> {
  return await requestJson(
    `/api/runtime/services/${encodeURIComponent(serviceId)}/state`,
    jsonRequest('POST', { nodeId, field, value }),
  ) as JsonValue;
}
