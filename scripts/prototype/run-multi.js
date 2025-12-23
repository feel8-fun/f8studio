#!/usr/bin/env node
/**
 * Multi-process prototype using IPC to mimic NATS + KV (memory only).
 * Processes: master, engine, web (client).
 * Subjects prefixed with f8.*; KV is per-bucket, revisioned (etag).
 */

const { fork } = require('child_process');
const path = require('path');

const roles = ['master', 'engine', 'web'];

if (process.env.ROLE) {
  runChild(process.env.ROLE);
} else {
  runOrchestrator();
}

function runOrchestrator() {
  const broker = {
    subs: new Map(), // subjectPattern -> Set(child)
    kv: new Map(), // bucket -> Map(key -> {value, rev})
  };
  const children = new Map();

  for (const role of roles) {
    const child = fork(__filename, [], { env: { ...process.env, ROLE: role }, stdio: ['inherit', 'inherit', 'inherit', 'ipc'] });
    children.set(child.pid, { child, role });
    child.on('message', (msg) => handleMessage(broker, children, child, msg));
    child.on('exit', () => {
      // If web exits, stop the rest to avoid hanging.
      if (role === 'web') {
        for (const { child: ch, role: r } of children.values()) {
          if (r !== 'web') ch.kill();
        }
        process.exit(0);
      }
    });
  }
}

function handleMessage(broker, children, child, msg) {
  if (!msg || typeof msg !== 'object') return;
  switch (msg.type) {
    case 'sub': {
      if (!broker.subs.has(msg.subject)) broker.subs.set(msg.subject, new Set());
      broker.subs.get(msg.subject).add(child);
      break;
    }
    case 'unsub': {
      broker.subs.get(msg.subject)?.delete(child);
      break;
    }
    case 'pub': {
      for (const [pattern, subs] of broker.subs.entries()) {
        if (subjectMatch(msg.subject, pattern)) {
          subs.forEach((ch) => ch.send({ type: 'pub', subject: msg.subject, data: msg.data }));
        }
      }
      break;
    }
    case 'kv.get': {
      const bucket = broker.kv.get(msg.bucket) || new Map();
      const entry = bucket.get(msg.key);
      child.send({ type: 'kv.resp', reqId: msg.reqId, value: entry?.value, rev: entry?.rev ?? 0, ok: true });
      break;
    }
    case 'kv.put': {
      const bucket = broker.kv.get(msg.bucket) || new Map();
      const entry = bucket.get(msg.key) || { rev: 0, value: null };
      const expected = msg.expected ?? entry.rev;
      if (expected !== entry.rev) {
        child.send({ type: 'kv.resp', reqId: msg.reqId, ok: false, error: 'BAD_REV', current: entry.rev });
        break;
      }
      const nextRev = entry.rev + 1;
      bucket.set(msg.key, { value: msg.value, rev: nextRev });
      broker.kv.set(msg.bucket, bucket);
      child.send({ type: 'kv.resp', reqId: msg.reqId, ok: true, rev: nextRev });
      break;
    }
    default:
      break;
  }
}

function subjectMatch(subject, pattern) {
  if (pattern.endsWith('*')) {
    const prefix = pattern.slice(0, -1);
    return subject.startsWith(prefix);
  }
  return subject === pattern;
}

async function runChild(role) {
  const { publish, subscribe, kvGet, kvPut, request } = require('./ipc-bus');
  const log = (msg) => console.log(`[${new Date().toISOString()}][${role}] ${msg}`);

  if (role === 'master') {
    let up = true;
    const bucket = 'kv_graph';
    const key = 'graph';
    subscribe('f8.master.ping', ({ reply }) => {
      publish(reply, up ? { status: 'ok' } : { status: 'unavailable' });
    });
    subscribe('f8.master.snapshot', async ({ reply }) => {
      if (!up) return publish(reply, { error: 'MASTER_UNAVAILABLE' });
      const res = await kvGet(bucket, key);
      publish(reply, { graph: res.value, etag: res.rev });
    });
    subscribe('f8.master.toggle', ({ up: newUp }) => {
      up = !!newUp;
      log(`master availability -> ${up}`);
    });
    subscribe('f8.master.apply', async ({ graph, expectedEtag, reply }) => {
      if (!up) return publish(reply, { error: 'MASTER_UNAVAILABLE', message: 'read-only; master down' });
      const put = await kvPut(bucket, key, graph, expectedEtag ?? 0);
      if (!put.ok) return publish(reply, { error: 'ETAG_MISMATCH', currentEtag: put.current });
      publish('f8.control.apply', { graph, etag: put.rev });
      publish(reply, { ok: true, etag: put.rev });
      log(`applied graph etag=${put.rev}`);
    });
    return;
  }

  if (role === 'engine') {
    const instanceId = 'engineA';
    let graph = null;
    const queues = new Map();
    subscribe('f8.control.apply', ({ graph: g, etag }) => {
      graph = g;
      queues.clear();
      if (graph?.edges) {
        for (const e of graph.edges) {
          if (e.scope === 'cross') {
            const subj = `f8.bus.${e.edgeId}`;
            queues.set(e.edgeId, []);
            subscribe(subj, (msg) => {
              const q = queues.get(e.edgeId);
              if (!q) return;
              q.length = 0;
              q.push(msg);
            });
          }
        }
      }
      log(`applied graph etag=${etag}`);
    });

    // Receive other instances' state (wildcard)
    subscribe('f8.state.', (msg, subject) => {
      if (!subject.startsWith('f8.state.') || subject.endsWith(instanceId + '.set')) return;
      log(`received state from ${subject}: ${JSON.stringify(msg)}`);
    });

    setInterval(() => tickEngine(), 300);

    function tickEngine() {
      for (const [edgeId, q] of queues.entries()) {
        if (q.length === 0) continue;
        const msg = q[q.length - 1];
        log(`consumed data on edge ${edgeId}: ${JSON.stringify(msg)}`);
        q.length = 0;
      }
      // broadcast local state
      publish(`f8.state.${instanceId}.set`, { state: { now: Date.now() } });
    }
    return;
  }

  if (role === 'web') {
    const logStep = (msg) => log(msg);
    const apply = (graph, expectedEtag) => request('f8.master.apply', { graph, expectedEtag });
    const ping = () => request('f8.master.ping', {});
    const toggle = (up) => publish('f8.master.toggle', { up });

    const graphV1 = { edges: [{ edgeId: 'edge1', scope: 'cross', strategy: 'latest', queueSize: 64, dropOld: true }] };
    const graphV2 = { edges: [{ edgeId: 'edge2', scope: 'cross', strategy: 'latest', queueSize: 64, dropOld: true }] };

    logStep('ping master');
    logStep(JSON.stringify(await ping()));
    logStep('apply graph v1');
    logStep(JSON.stringify(await apply(graphV1, 0)));

    // publish some data
    publish('f8.bus.edge1', { payload: { value: 1 }, ts: Date.now() });

    setTimeout(async () => {
      logStep('master down toggle');
      toggle(false);
      logStep('apply graph v2 (expect read-only error)');
      logStep(JSON.stringify(await apply(graphV2, 1)));
      publish('f8.bus.edge1', { payload: { value: 99 }, ts: Date.now() });
    }, 500);

    setTimeout(async () => {
      logStep('master up toggle');
      toggle(true);
      logStep('apply graph v2 (expect success)');
      logStep(JSON.stringify(await apply(graphV2, 1)));
      publish('f8.bus.edge2', { payload: { value: 7 }, ts: Date.now() });
    }, 1000);

    setTimeout(() => {
      logStep('demo done, exiting');
      process.exit(0);
    }, 1600);
  }
}
