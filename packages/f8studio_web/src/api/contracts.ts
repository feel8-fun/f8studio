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
