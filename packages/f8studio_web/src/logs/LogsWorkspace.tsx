import { CircleDot, RefreshCw, Search } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { fetchLogs } from '../api/client';
import { isStudioLogEvent, type JsonValue, type StudioLogEvent } from '../api/contracts';

type LogLevel = 'info' | 'warning' | 'error';
type LevelFilter = 'all' | LogLevel;

interface DisplayLog {
  readonly level: LogLevel;
  readonly source: string;
  readonly message: string;
  readonly details: readonly string[];
}

function isLogType(type: string): boolean {
  return type === 'service.log' || type.startsWith('deploy.') ||
    type.startsWith('service.process_') || type === 'runtime.error' ||
    type === 'media.error' || type === 'server.error';
}

function logPayload(value: JsonValue): Readonly<Record<string, JsonValue>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Readonly<Record<string, JsonValue>> : {};
}

function stringField(payload: Readonly<Record<string, JsonValue>>, field: string): string {
  const value = payload[field];
  return typeof value === 'string' ? value : '';
}

function displayLog(event: StudioLogEvent): DisplayLog {
  const payload = logPayload(event.payload);
  const serviceId = stringField(payload, 'serviceId');
  const source = serviceId || event.scope.replace(/^(project|service):/, '');
  if (event.type === 'service.log') {
    const message = stringField(payload, 'line').trim();
    const level: LogLevel = /\b(ERROR|CRITICAL|FATAL)\b/i.test(message) ? 'error'
      : /\bWARN(?:ING)?\b/i.test(message) ? 'warning' : 'info';
    return { level, source, message, details: [] };
  }
  if (event.type === 'deploy.finished') {
    const status = stringField(payload, 'status');
    const revision = payload.sourceGraphRevision;
    const details = Array.isArray(payload.serviceResults)
      ? payload.serviceResults.flatMap((item) => {
        const result = logPayload(item);
        const id = stringField(result, 'serviceId');
        const error = stringField(result, 'errorMessage');
        return error ? [`${id}: ${error}`] : [];
      }) : [];
    const error = stringField(payload, 'errorMessage');
    if (error) details.unshift(error);
    return {
      level: status === 'succeeded' ? 'info' : 'error', source,
      message: `Deployment ${status || 'finished'}${typeof revision === 'number' ? ` · r${revision}` : ''}`,
      details,
    };
  }
  if (event.type.startsWith('deploy.')) {
    const revision = payload.sourceGraphRevision;
    return { level: 'info', source, message: `Deployment ${event.type.slice(7)}${typeof revision === 'number' ? ` · r${revision}` : ''}`, details: [] };
  }
  if (event.type === 'service.process_started') {
    return { level: 'info', source, message: `Process started · ${stringField(payload, 'serviceClass')}`, details: [] };
  }
  if (event.type === 'service.process_stopped') {
    return { level: 'info', source, message: 'Process stopped', details: [] };
  }
  return {
    level: 'error', source,
    message: [stringField(payload, 'operation'), stringField(payload, 'message')].filter(Boolean).join(': '),
    details: [],
  };
}

function mergeLogs(current: readonly StudioLogEvent[], incoming: readonly StudioLogEvent[]): readonly StudioLogEvent[] {
  if (incoming.length === 0) return current;
  const epoch = incoming[incoming.length - 1]?.serverEpoch;
  const merged = new Map(current.filter((event) => event.serverEpoch === epoch).map((event) => [event.eventId, event]));
  for (const event of incoming) merged.set(event.eventId, event);
  return [...merged.values()].sort((left, right) => left.sequence - right.sequence).slice(-1000);
}

export function LogsWorkspace() {
  const [events, setEvents] = useState<readonly StudioLogEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [level, setLevel] = useState<LevelFilter>('all');
  const [follow, setFollow] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const snapshot = await fetchLogs(signal);
      setEvents((current) => mergeLogs(current, snapshot));
      setError(null);
    } catch (reason: unknown) {
      if (signal?.aborted) return;
      setError(reason instanceof Error ? reason.message : 'Unable to load logs');
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let socket: WebSocket | null = null;
    let retryTimer: number | null = null;
    let retry = 0;
    const connect = () => {
      if (controller.signal.aborted) return;
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/events`);
      socket.onopen = () => {
        retry = 0;
        setConnected(true);
        void load(controller.signal);
      };
      socket.onmessage = (message) => {
        let decoded: unknown;
        try {
          decoded = JSON.parse(String(message.data));
        } catch (reason: unknown) {
          console.error('Invalid log event JSON', reason);
          return;
        }
        if (isStudioLogEvent(decoded) && isLogType(decoded.type)) {
          setEvents((current) => mergeLogs(current, [decoded]));
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!controller.signal.aborted) retryTimer = window.setTimeout(connect, Math.min(5000, 300 * 2 ** retry++));
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => {
      controller.abort();
      socket?.close();
      if (retryTimer !== null) window.clearTimeout(retryTimer);
    };
  }, [load]);

  const visible = useMemo(() => events.flatMap((event) => {
    const row = displayLog(event);
    if (level !== 'all' && row.level !== level) return [];
    const needle = query.trim().toLowerCase();
    if (needle && !`${row.source} ${row.message} ${row.details.join(' ')}`.toLowerCase().includes(needle)) return [];
    return [{ event, row }];
  }), [events, level, query]);

  useEffect(() => {
    if (follow && listRef.current !== null) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [follow, visible]);

  return <section className="logs-workspace" aria-label="Log center">
    <div className="logs-toolbar">
      <label className="logs-search"><Search size={15} /><input aria-label="Search logs" placeholder="Search logs" value={query}
        onChange={(event) => setQuery(event.target.value)} /></label>
      <select aria-label="Log level" value={level} onChange={(event) => setLevel(event.target.value as LevelFilter)}>
        <option value="all">All levels</option><option value="error">Errors</option>
        <option value="warning">Warnings</option><option value="info">Info</option>
      </select>
      <label className="logs-follow"><input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />Follow</label>
      <span className={`logs-connection ${connected ? 'online' : ''}`}><CircleDot size={13} />{connected ? 'Live' : 'Reconnecting'}</span>
      <button type="button" className="icon-button bordered" aria-label="Refresh logs" title="Refresh logs" onClick={() => void load()}><RefreshCw size={15} /></button>
    </div>
    {error !== null && <p className="logs-error" role="alert">{error}</p>}
    <div className="logs-list" ref={listRef} role="log" aria-live="off">
      {visible.map(({ event, row }) => <div className="logs-row" data-level={row.level} key={event.eventId}>
        <time dateTime={event.timestamp}>{new Date(event.timestamp).toLocaleTimeString()}</time>
        <span className="logs-level">{row.level}</span>
        <span className="logs-source" title={row.source}>{row.source}</span>
        <div className="logs-message">{row.message}{row.details.map((detail, index) => <small key={`${event.eventId}:${index}`}>{detail}</small>)}</div>
      </div>)}
      {visible.length === 0 && <div className="empty-state">{events.length === 0 ? 'No logs yet' : 'No matching logs'}</div>}
    </div>
  </section>;
}
