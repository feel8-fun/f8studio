import type { ProjectSummary, AgentImage, EditorAnalysis, EditorDiagnostic, RuntimeActionResult, RuntimeNodeState, RuntimeStateField, MediaSessionAnswer, AudioSessionAnswer, ServiceDeployResult, DeployJob } from './contracts.gen';
export type { ProjectSummary, AgentImage, EditorAnalysis, EditorDiagnostic, RuntimeActionResult, RuntimeNodeState, RuntimeStateField, MediaSessionAnswer, AudioSessionAnswer, ServiceDeployResult, DeployJob } from './contracts.gen';
export type HealthStatus = import('./contracts.gen').HealthStatus;

export type ServerCapabilities = import('./contracts.gen').ServerCapabilities;

export type CapabilitiesResponse = import('./contracts.gen').CapabilitiesResponse;

export type JsonValue = import('./contracts.gen').JsonValue;

export function isJsonObject(value: unknown): value is Readonly<Record<string, JsonValue>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export type StudioLogEvent = import('./contracts.gen').EventEnvelope;

export function isStudioLogEvent(value: unknown): value is StudioLogEvent {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const event = value as Record<string, unknown>;
  return typeof event.eventId === 'string' && typeof event.serverEpoch === 'string' &&
    typeof event.sequence === 'number' && typeof event.type === 'string' &&
    typeof event.scope === 'string' && typeof event.timestamp === 'string' &&
    'payload' in event;
}

export type AgentRunStatus = import('./contracts.gen').AgentRunStatus;
export type AgentToolCallStatus = import('./contracts.gen').ToolCallStatus;
export type AgentApprovalStatus = import('./contracts.gen').ApprovalStatus;

export type AgentProviderSummary = import('./contracts.gen').AgentProviderSummary;

export type AgentModelCapabilities = import('./contracts.gen').ModelCapabilitiesInput;

export type AgentProviderSettings = import('./contracts.gen').ProviderSettingsView;

export type UpdateAgentProviderSettings = import('./contracts.gen').UpdateProviderSettingsInput;

export type CreateAgentConnection = import('./contracts.gen').CreateProviderConnectionInput;

export type AgentConnectionProbe = import('./contracts.gen').ProviderProbeResult;

export type AgentMessage = import('./contracts.gen').AgentMessage;

export type AgentToolCall = import('./contracts.gen').AgentToolCall;

export type AgentApproval = import('./contracts.gen').AgentApproval;

export type AgentArtifact = import('./contracts.gen').AgentArtifact;

export type AgentSessionSummary = import('./contracts.gen').AgentSessionSummary;

export type AgentSession = import('./contracts.gen').AgentSessionRecord;

export type ValueSchema = StateSpec['valueSchema'];

export type StateSpec = import('./contracts.gen').F8StateSpec;

export type UiControlSpec = import('./contracts.gen').F8UiControlSpec;

export type CollectionEditPolicy = import('./contracts.gen').F8CollectionEditPolicy;

export type SpecEditPolicy = import('./contracts.gen').F8SpecEditPolicy;

export type DataPortSpec = import('./contracts.gen').F8DataPortSpec;

export type CommandParamSpec = import('./contracts.gen').F8CommandParam;

export type CommandSpec = import('./contracts.gen').F8Command;

export type ServiceSpec = import('./contracts.gen').F8ServiceSpec;

export type OperatorSpec = import('./contracts.gen').F8OperatorSpec;

export type ExecPortSpec = import('./contracts.gen').F8ExecPortSpec;

export type CatalogSnapshot = import('./contracts.gen').CatalogSnapshot;
export type ExtensionStatus = import('./contracts.gen').ExtensionStatus;
export type EnvironmentStatus = import('./contracts.gen').EnvironmentStatus;

export type NodeKind = GraphNode['kind'];
export type PortKind = import('./contracts.gen').PortKind;
export type PortDirection = import('./contracts.gen').PortDirection;
export type GraphEdgeKind = import('./contracts.gen').GraphEdgeKind;

export type GraphPort = import('./contracts.gen').GraphPort;

export type ServiceNode = import('./contracts.gen').ServiceNode;

export type OperatorNode = import('./contracts.gen').OperatorNode;

export type GraphNode = ServiceNode | OperatorNode;

export type GraphEdge = import('./contracts.gen').GraphEdge;

export type NodeLayout = import('./contracts.gen').NodeLayout;

export type StudioDocument = import('./contracts.gen').StudioDocument;

export type ProjectRecord = import('./contracts.gen').ProjectRecord;

export type GraphOperation = import('./contracts.gen').PatchRequestInput['operations'][number];

export type PatchResult = import('./contracts.gen').PatchResult;

export type DeployJobStatus = import('./contracts.gen').JobStatus;

export type RuntimeMonitor = import('./contracts.gen').F8MonitorSnapshot;

export function isDeployJob(value: unknown): value is DeployJob {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return typeof item.jobId === 'string' && typeof item.projectId === 'string' &&
    typeof item.sourceGraphRevision === 'number' && typeof item.status === 'string' &&
    Array.isArray(item.serviceResults);
}

export function isStudioDocument(value: unknown): value is StudioDocument {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return item.schemaVersion === 'f8studio-document/2' && typeof item.projectId === 'string' &&
    typeof item.graphRevision === 'number' && typeof item.layoutRevision === 'number' &&
    Array.isArray(item.nodes) && Array.isArray(item.edges) && Array.isArray(item.layout);
}

export function isProjectRecord(value: unknown): value is ProjectRecord {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return typeof item.projectId === 'string' && typeof item.name === 'string' && isStudioDocument(item.document);
}

export function isGraphNode(value: unknown): value is GraphNode {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return (item.kind === 'service' || item.kind === 'operator') && typeof item.nodeId === 'string' &&
    typeof item.name === 'string' && typeof item.serviceClass === 'string' && Array.isArray(item.ports);
}

export type RtcIceServer = import('./contracts.gen').BrowserIceServer;

export type RtcConfigurationResponse = import('./contracts.gen').BrowserRtcConfiguration;

export type AssetKind = import('./contracts.gen').AssetKind;

export type AssetSummary = import('./contracts.gen').AssetSummary;

export type AssetRecord = import('./contracts.gen').AssetRecord;

export type AssetVersion = import('./contracts.gen').AssetVersion;

export type ProjectVersion = import('./contracts.gen').ProjectVersion;

export type EditorSession = import('./contracts.gen').EditorSessionRecord;

export type EditorLanguageResult = import('./contracts.gen').EditorLanguageResult;

export type LocalCapability = import('./contracts.gen').LocalCapability;

export type SerialPortInfo = import('./contracts.gen').SerialPortInfo;

export type UnityInstallPlan = import('./contracts.gen').UnityInstallPlan;

export type SkeletonUdpVerification = import('./contracts.gen').SkeletonUdpVerification;

export type HotkeyBinding = import('./contracts.gen').HotkeyBinding;

export type RegisterHotkeyInput = import('./contracts.gen').RegisterHotkeyRequestInput;

export type PresentationCommand = import('./contracts.gen').PresentationCommand;

export type SkeletonNode = import('./contracts.gen').SkeletonNode;
export type SkeletonPerson = import('./contracts.gen').SkeletonPerson;
export type SkeletonScene = import('./contracts.gen').SkeletonScene;

export function isHealthStatus(value: unknown): value is HealthStatus {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return (
    item.status === 'ok' &&
    typeof item.service === 'string' &&
    typeof item.version === 'string' &&
    item.protocol_version === 'f8studio-api/1' &&
    typeof item.server_epoch === 'string' &&
    item.server_epoch.length > 0
  );
}

export function isMediaSessionAnswer(value: unknown): value is MediaSessionAnswer {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return (
    typeof item.sessionId === 'string' &&
    typeof item.source === 'string' &&
    (item.quality === 'thumbnail' || item.quality === 'main') &&
    typeof item.sdp === 'string' &&
    item.type === 'answer' &&
    typeof item.maxWidth === 'number' &&
    typeof item.maxHeight === 'number' &&
    typeof item.maxFps === 'number' &&
    typeof item.overlay === 'boolean'
  );
}

export function isAudioSessionAnswer(value: unknown): value is AudioSessionAnswer {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  return (
    typeof item.sessionId === 'string' &&
    typeof item.source === 'string' &&
    typeof item.sdp === 'string' &&
    item.type === 'answer' &&
    typeof item.sampleRate === 'number' &&
    typeof item.channels === 'number' &&
    typeof item.transportPolicy === 'string'
  );
}

export function isRtcConfigurationResponse(value: unknown): value is RtcConfigurationResponse {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  if (item.iceTransportPolicy !== 'all' && item.iceTransportPolicy !== 'relay') return false;
  if (!Array.isArray(item.iceServers)) return false;
  return item.iceServers.every((serverValue) => {
    if (typeof serverValue !== 'object' || serverValue === null) return false;
    const server = serverValue as Record<string, unknown>;
    return (
      Array.isArray(server.urls) &&
      server.urls.length > 0 &&
      server.urls.every((url) => typeof url === 'string' && url.length > 0) &&
      (server.username === undefined || typeof server.username === 'string') &&
      (server.credential === undefined || typeof server.credential === 'string')
    );
  });
}

function isVec3(value: unknown): value is readonly [number, number, number] {
  return Array.isArray(value) && value.length === 3 && value.every((item) => typeof item === 'number');
}

function isSkeletonNode(value: unknown): value is SkeletonNode {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  const validRotation = item.rot === null || (
    Array.isArray(item.rot) && item.rot.length === 4 && item.rot.every((component) => typeof component === 'number')
  );
  return typeof item.index === 'number' && typeof item.name === 'string' && isVec3(item.pos) && validRotation;
}

function isSkeletonEdge(value: unknown): value is readonly [number, number] {
  return Array.isArray(value) && value.length === 2 && value.every((item) => Number.isInteger(item));
}

export function isSkeletonScene(value: unknown): value is SkeletonScene {
  if (typeof value !== 'object' || value === null) return false;
  const scene = value as Record<string, unknown>;
  if (typeof scene.tsMs !== 'number' || typeof scene.worldUp !== 'string' || !Array.isArray(scene.people)) return false;
  return scene.people.every((personValue) => {
    if (typeof personValue !== 'object' || personValue === null) return false;
    const person = personValue as Record<string, unknown>;
    const validBox = person.bbox === null || (
      Array.isArray(person.bbox) && person.bbox.every((item) => typeof item === 'number')
    );
    const validEdges = person.skeletonEdges === null || (
      Array.isArray(person.skeletonEdges) && person.skeletonEdges.every(isSkeletonEdge)
    );
    return (
      typeof person.name === 'string' &&
      typeof person.skeletonProtocol === 'string' &&
      validBox &&
      validEdges &&
      Array.isArray(person.nodes) &&
      person.nodes.every(isSkeletonNode)
    );
  });
}
