export interface HealthStatus {
  readonly status: 'ok';
  readonly service: string;
  readonly version: string;
  readonly protocol_version: 'f8studio-api/1';
  readonly server_epoch: string;
}

export interface ServerCapabilities {
  readonly graph_editing: boolean;
  readonly runtime_control: boolean;
  readonly web_assets: boolean;
  readonly web_rtc_video: boolean;
  readonly web_rtc_audio: boolean;
  readonly three_d: boolean;
  readonly agent_tools: boolean;
}

export interface CapabilitiesResponse {
  readonly protocol_version: 'f8studio-api/1';
  readonly capabilities: ServerCapabilities;
}

export type JsonValue = null | boolean | number | string | readonly JsonValue[] | { readonly [key: string]: JsonValue };

export interface ValueSchema {
  readonly type?: string;
  readonly default?: JsonValue;
  readonly enum?: readonly JsonValue[];
  readonly minimum?: number;
  readonly maximum?: number;
  readonly exclusiveMinimum?: number;
  readonly exclusiveMaximum?: number;
  readonly multipleOf?: number;
  readonly required?: readonly string[];
  readonly additionalProperties?: boolean;
  readonly [key: string]: JsonValue | undefined;
}

export interface StateSpec {
  readonly name: string;
  readonly valueSchema: ValueSchema;
  readonly access: 'rw' | 'ro' | 'wo';
  readonly label?: string;
  readonly description?: string;
  readonly showOnNode?: boolean;
  readonly required?: boolean;
  readonly uiControl?: string;
  readonly redactOnPublish?: boolean;
}

export interface DataPortSpec {
  readonly name: string;
  readonly valueSchema: ValueSchema;
  readonly payloadKind?: string;
  readonly delivery?: string;
  readonly payload?: { readonly kind: string; readonly [key: string]: JsonValue };
  readonly showOnNode?: boolean;
  readonly [key: string]: JsonValue | ValueSchema | undefined;
}

export interface ServiceSpec {
  readonly serviceClass: string;
  readonly label: string;
  readonly specKind: 'service';
  readonly description?: string;
  readonly tags?: readonly string[];
  readonly paletteCategory?: string;
  readonly hiddenInPalette?: boolean;
  readonly stateFields?: readonly StateSpec[];
  readonly dataInPorts?: readonly DataPortSpec[];
  readonly dataOutPorts?: readonly DataPortSpec[];
  readonly [key: string]: JsonValue | readonly StateSpec[] | readonly DataPortSpec[] | undefined;
}

export interface OperatorSpec {
  readonly operatorClass: string;
  readonly serviceClass: string;
  readonly label: string;
  readonly specKind: 'operator';
  readonly description?: string;
  readonly tags?: readonly string[];
  readonly paletteCategory?: string;
  readonly hiddenInPalette?: boolean;
  readonly stateFields?: readonly StateSpec[];
  readonly dataInPorts?: readonly DataPortSpec[];
  readonly dataOutPorts?: readonly DataPortSpec[];
  readonly execInPorts?: readonly string[];
  readonly execOutPorts?: readonly string[];
  readonly [key: string]: JsonValue | readonly StateSpec[] | readonly DataPortSpec[] | undefined;
}

export interface CatalogSnapshot {
  readonly services: readonly ServiceSpec[];
  readonly operators: readonly OperatorSpec[];
}

export type NodeKind = 'service' | 'operator';
export type PortKind = 'data' | 'state' | 'exec' | 'command';
export type PortDirection = 'input' | 'output';
export type GraphEdgeKind = 'data' | 'state' | 'exec';

export interface GraphPort {
  readonly portId: string;
  readonly name: string;
  readonly runtimeName: string;
  readonly kind: PortKind;
  readonly direction: PortDirection;
  readonly dataSpec?: DataPortSpec | null;
  readonly stateSpec?: StateSpec | null;
}

interface GraphNodeBase {
  readonly nodeId: string;
  readonly name: string;
  readonly serviceId: string;
  readonly serviceClass: string;
  readonly ports: readonly GraphPort[];
  readonly stateValues: Readonly<Record<string, JsonValue>>;
  readonly enabled: boolean;
}

export interface ServiceNode extends GraphNodeBase {
  readonly kind: 'service';
  readonly spec: ServiceSpec;
}

export interface OperatorNode extends GraphNodeBase {
  readonly kind: 'operator';
  readonly operatorClass: string;
  readonly spec: OperatorSpec;
}

export type GraphNode = ServiceNode | OperatorNode;

export interface GraphEdge {
  readonly edgeId: string;
  readonly fromNodeId: string;
  readonly fromPortId: string;
  readonly toNodeId: string;
  readonly toPortId: string;
  readonly kind: GraphEdgeKind;
  readonly strategy: 'latest' | 'queue';
  readonly queueSize: number;
  readonly timeoutMs: number | null;
}

export interface NodeLayout {
  readonly nodeId: string;
  readonly x: number;
  readonly y: number;
  readonly width?: number | null;
  readonly height?: number | null;
  readonly collapsed: boolean;
}

export interface StudioDocument {
  readonly schemaVersion: 'f8studio-document/1';
  readonly projectId: string;
  readonly graphId: string;
  readonly graphRevision: number;
  readonly layoutRevision: number;
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
  readonly layout: readonly NodeLayout[];
}

export interface ProjectSummary {
  readonly projectId: string;
  readonly name: string;
  readonly description: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly graphRevision: number;
  readonly layoutRevision: number;
}

export interface ProjectRecord {
  readonly projectId: string;
  readonly name: string;
  readonly description: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly document: StudioDocument;
}

export type GraphOperation =
  | { readonly op: 'createNode'; readonly node: GraphNode; readonly layout?: NodeLayout | null }
  | { readonly op: 'deleteNode'; readonly nodeId: string }
  | { readonly op: 'connectEdge'; readonly edge: GraphEdge }
  | { readonly op: 'disconnectEdge'; readonly edgeId: string }
  | { readonly op: 'setNodeLayout'; readonly layout: NodeLayout }
  | { readonly op: 'renameNode'; readonly nodeId: string; readonly name: string }
  | { readonly op: 'bindOperatorService'; readonly nodeId: string; readonly serviceId: string }
  | { readonly op: 'setNodeEnabled'; readonly nodeId: string; readonly enabled: boolean }
  | { readonly op: 'setNodeState'; readonly nodeId: string; readonly field: string; readonly value: JsonValue }
  | { readonly op: 'insertFragment'; readonly nodes: readonly GraphNode[]; readonly edges: readonly GraphEdge[]; readonly layout: readonly NodeLayout[] };

export interface PatchResult {
  readonly requestId: string;
  readonly graphChanged: boolean;
  readonly layoutChanged: boolean;
  readonly document: StudioDocument;
}

export type DeployJobStatus = 'queued' | 'running' | 'succeeded' | 'partially_failed' | 'failed' | 'cancelled';

export interface ServiceDeployResult {
  readonly serviceId: string;
  readonly success: boolean;
  readonly errorMessage: string;
}

export interface DeployJob {
  readonly jobId: string;
  readonly requestId: string;
  readonly projectId: string;
  readonly sourceGraphRevision: number;
  readonly sourceSemanticRevision: string;
  readonly status: DeployJobStatus;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly serviceResults: readonly ServiceDeployResult[];
  readonly errorMessage: string;
}

export interface RuntimeMonitor {
  readonly serviceId: string;
  readonly serviceClass: string;
  readonly nodeId: string;
  readonly tsMs: number;
  readonly alive: boolean;
  readonly ready: boolean;
  readonly active: boolean;
  readonly uptimeMs: number;
  readonly cpu?: { readonly processPercent?: number; readonly systemPercent?: number };
  readonly memory?: { readonly rssBytes?: number; readonly vmsBytes?: number };
  readonly queue?: { readonly depth?: number };
  readonly timing?: { readonly processMsP95?: number; readonly latencyMsP95?: number };
  readonly error?: { readonly currentMessage?: string; readonly lastMessage?: string };
}

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
  return item.schemaVersion === 'f8studio-document/1' && typeof item.projectId === 'string' &&
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

export interface MediaSessionAnswer {
  readonly sessionId: string;
  readonly source: string;
  readonly quality: 'thumbnail' | 'main';
  readonly sdp: string;
  readonly type: 'answer';
  readonly maxWidth: number;
  readonly maxHeight: number;
  readonly maxFps: number;
  readonly overlay: boolean;
}

export interface AudioSessionAnswer {
  readonly sessionId: string;
  readonly source: string;
  readonly sdp: string;
  readonly type: 'answer';
  readonly sampleRate: number;
  readonly channels: number;
  readonly transportPolicy: string;
}

export interface RtcIceServer {
  readonly urls: readonly string[];
  readonly username?: string;
  readonly credential?: string;
}

export interface RtcConfigurationResponse {
  readonly iceServers: readonly RtcIceServer[];
  readonly iceTransportPolicy: 'all' | 'relay';
}

export interface MediaSample {
  readonly source: string;
  readonly format: string;
  readonly frameId: number;
  readonly tsMs: number;
  readonly width: number;
  readonly height: number;
  readonly x: number;
  readonly y: number;
  readonly finite: boolean;
  readonly value: number | null | Readonly<Record<string, number | null>>;
  readonly streamId: string;
  readonly streamEpoch: string;
}

export interface SkeletonNode {
  readonly index: number;
  readonly name: string;
  readonly pos: readonly [number, number, number];
  readonly rot: readonly [number, number, number, number] | null;
}

export interface SkeletonPerson {
  readonly name: string;
  readonly bbox: readonly number[] | null;
  readonly skeletonProtocol: string;
  readonly skeletonEdges: readonly (readonly [number, number])[] | null;
  readonly nodes: readonly SkeletonNode[];
}

export interface SkeletonScene {
  readonly tsMs: number;
  readonly worldUp: string;
  readonly people: readonly SkeletonPerson[];
}

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

export function isMediaSample(value: unknown): value is MediaSample {
  if (typeof value !== 'object' || value === null) return false;
  const item = value as Record<string, unknown>;
  const validValue = item.value === null || typeof item.value === 'number' || (
    typeof item.value === 'object' &&
    item.value !== null &&
    Object.values(item.value).every((component) => component === null || typeof component === 'number')
  );
  return (
    typeof item.source === 'string' &&
    typeof item.format === 'string' &&
    typeof item.frameId === 'number' &&
    typeof item.tsMs === 'number' &&
    typeof item.width === 'number' &&
    typeof item.height === 'number' &&
    typeof item.x === 'number' &&
    typeof item.y === 'number' &&
    typeof item.finite === 'boolean' &&
    typeof item.streamId === 'string' &&
    typeof item.streamEpoch === 'string' &&
    validValue
  );
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
