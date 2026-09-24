import { CircleDot, RefreshCw, Search } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { fetchLogs } from '../api/client';
import { isStudioLogEvent, type JsonValue, type StudioLogEvent } from '../api/contracts';

type LogLevel = 'info' | 'warning' | 'error';
type LevelFilter = 'all' | LogLevel;
const LOG_PAGE_SIZE = 100;
const MAX_LOADED_LOGS = 300;

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

function mergeLogs(current: readonly StudioLogEvent[], incoming: readonly StudioLogEvent[], keep: 'latest' | 'oldest' = 'latest'): readonly StudioLogEvent[] {
  if (incoming.length === 0) return current;
  const epoch = incoming[incoming.length - 1]?.serverEpoch;
  const merged = new Map(current.filter((event) => event.serverEpoch === epoch).map((event) => [event.eventId, event]));
  for (const event of incoming) merged.set(event.eventId, event);
  const ordered = [...merged.values()].sort((left, right) => left.sequence - right.sequence);
  return keep === 'oldest' ? ordered.slice(0, MAX_LOADED_LOGS) : ordered.slice(-MAX_LOADED_LOGS);
}

export function LogsWorkspace() {
  const [events, setEvents] = useState<readonly StudioLogEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [level, setLevel] = useState<LevelFilter>('all');
  const [follow, setFollow] = useState(true);
  const [historyMode, setHistoryMode] = useState(false);
  const [hasOlder, setHasOlder] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const historyModeRef = useRef(false);
  const listRef = useRef<HTMLDivElement>(null);

  const loadLatest = useCallback(async (signal?: AbortSignal) => {
    try {
      const snapshot = await fetchLogs({ limit: LOG_PAGE_SIZE, signal });
      setEvents((current) => mergeLogs(current, snapshot));
      setHasOlder(snapshot.length === LOG_PAGE_SIZE);
      setError(null);
    } catch (reason: unknown) {
      if (signal?.aborted) return;
      setError(reason instanceof Error ? reason.message : 'Unable to load logs');
    }
  }, []);

  const loadOlder = async () => {
    const beforeSequence = events[0]?.sequence;
    if (beforeSequence === undefined || loadingOlder) return;
    historyModeRef.current = true;
    setHistoryMode(true);
    setFollow(false);
    setLoadingOlder(true);
    try {
      const older = await fetchLogs({ limit: LOG_PAGE_SIZE, beforeSequence });
      setEvents((current) => mergeLogs(current, older, 'oldest'));
      setHasOlder(older.length === LOG_PAGE_SIZE);
      setError(null);
      if (listRef.current !== null) listRef.current.scrollTop = 0;
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : 'Unable to load older logs');
    } finally {
      setLoadingOlder(false);
    }
  };

  const jumpLatest = () => {
    historyModeRef.current = false;
    setHistoryMode(false);
    setFollow(true);
    setEvents([]);
    void loadLatest();
  };

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
        if (!historyModeRef.current) void loadLatest(controller.signal);
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
          if (!historyModeRef.current) setEvents((current) => mergeLogs(current, [decoded]));
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
  }, [loadLatest]);

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
      <button type="button" className="icon-button bordered" aria-label="Refresh logs" title="Refresh logs" onClick={jumpLatest}><RefreshCw size={15} /></button>
    </div>
    {error !== null && <p className="logs-error" role="alert">{error}</p>}
    {(hasOlder || historyMode) && <div className="logs-history-actions">
      {hasOlder && <button type="button" disabled={loadingOlder} onClick={() => void loadOlder()}>{loadingOlder ? 'Loading...' : 'Load older'}</button>}
      {historyMode && <button type="button" onClick={jumpLatest}>Latest</button>}
    </div>}
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
