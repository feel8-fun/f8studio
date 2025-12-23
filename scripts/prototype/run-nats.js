#!/usr/bin/env node
/**
 * Multi-process prototype using real nats.js + JetStream KV (memory only).
 * Subjects/KV keys follow f8.* prefix. Spawns: master, engineA, engineB, web.
 *
 * Prereq: run a local NATS with JetStream enabled, e.g.
 *   nats-server -js
 *
 * Run demo:
 *   pnpm proto:nats
 */

const { fork } = require('child_process');
const path = require('path');

const roles = [
  { role: 'master' },
  { role: 'engine', instance: 'engineA' },
  { role: 'engine', instance: 'engineB' },
  { role: 'web' },
];

if (process.env.ROLE) {
  runChild(process.env.ROLE, process.env.INSTANCE || '');
} else {
  runOrchestrator();
}

function runOrchestrator() {
  const children = [];
  for (const { role, instance } of roles) {
    const child = fork(path.join(__dirname, 'run-nats.js'), [], {
      env: { ...process.env, ROLE: role, INSTANCE: instance || '' },
      stdio: ['inherit', 'inherit', 'inherit', 'ipc'],
    });
    children.push({ child, role });
    child.on('exit', () => {
      if (role === 'web') {
        children.forEach(({ child: ch, role: r }) => {
          if (r !== 'web') ch.kill();
        });
      }
    });
  }
}

async function runChild(role, instanceId) {
  const { connect, StringCodec } = require('nats');
  const sc = StringCodec();
  const url = process.env.NATS_URL || 'nats://127.0.0.1:4222';
  const nc = await connect({ servers: url, name: `${role}${instanceId ? ':' + instanceId : ''}` });
  const js = nc.jetstream();
  const jsm = await nc.jetstreamManager();

  const log = (msg) => console.log(`[${new Date().toISOString()}][${role}${instanceId ? ':' + instanceId : ''}] ${msg}`);

  async function ensureKV(bucket) {
    try {
      return await js.views.kv(bucket); // bind existing
    } catch {
      // create new in-memory bucket
      return await js.views.kv({ bucket, storage: 'memory', description: 'proto KV' });
    }
  }

  if (role === 'master') {
    const kvName = 'kv_graph';
    const kv = await ensureKV(kvName);
    let up = true;

    // ping
    (async () => {
      const sub = nc.subscribe('f8.master.ping');
      for await (const m of sub) {
        m.respond(sc.encode(JSON.stringify(up ? { status: 'ok' } : { status: 'unavailable' })));
      }
    })();

    // snapshot
    (async () => {
      const sub = nc.subscribe('f8.master.snapshot');
      for await (const m of sub) {
        if (!up) {
          m.respond(sc.encode(JSON.stringify({ error: 'MASTER_UNAVAILABLE' })));
          continue;
        }
        const entry = await kv.get('graph').catch(() => null);
        const graph = entry && entry.data ? JSON.parse(sc.decode(entry.data)) : null;
        m.respond(sc.encode(JSON.stringify({ graph, etag: entry?.revision || 0 })));
      }
    })();

    // toggle availability
    (async () => {
      const sub = nc.subscribe('f8.master.toggle');
      for await (const m of sub) {
        const data = m.data.length ? JSON.parse(sc.decode(m.data)) : {};
        up = !!data.up;
        log(`master availability -> ${up}`);
      }
    })();

    // apply graph
    (async () => {
      const sub = nc.subscribe('f8.master.apply');
      for await (const m of sub) {
        if (!up) {
          m.respond(sc.encode(JSON.stringify({ error: 'MASTER_UNAVAILABLE', message: 'read-only; master down' })));
          continue;
        }
        const data = m.data.length ? JSON.parse(sc.decode(m.data)) : {};
        const graph = data.graph;
        const expected = data.expectedEtag;
        const entry = await kv.get('graph').catch(() => null);
        const current = entry?.revision || 0;
        if (typeof expected === 'number' && expected !== current) {
          m.respond(sc.encode(JSON.stringify({ error: 'ETAG_MISMATCH', currentEtag: current })));
          continue;
        }
        try {
          const rev = await kv.put('graph', sc.encode(JSON.stringify(graph)), { previousSeq: current });
          nc.publish('f8.control.apply', sc.encode(JSON.stringify({ graph, etag: rev })));
          m.respond(sc.encode(JSON.stringify({ ok: true, etag: rev })));
          log(`applied graph etag=${rev}`);
        } catch (err) {
          const entry2 = await kv.get('graph').catch(() => null);
          const current2 = entry2?.revision || 0;
          m.respond(sc.encode(JSON.stringify({ error: 'INTERNAL', currentEtag: current2, detail: err.message })));
        }
      }
    })();

    // Cleanup on exit: delete the graph bucket (best effort)
    process.on('exit', async () => {
      try {
        await (await nc.jetstreamManager()).kv.delete(kvName);
      } catch (_) {
        /* ignore */
      }
      try {
        await nc.close();
      } catch (_) {
        /* ignore */
      }
    });

    return;
  }

  if (role === 'engine') {
    const id = instanceId || `engine-${Math.random().toString(16).slice(2, 6)}`;
    let graph = null;
    const queues = new Map();
    const kvState = await ensureKV('kv_state');

    // Apply handler
    (async () => {
      const sub = nc.subscribe('f8.control.apply');
      for await (const m of sub) {
        const payload = JSON.parse(sc.decode(m.data));
        graph = payload.graph;
        queues.clear();
        if (graph?.edges) {
          for (const e of graph.edges) {
            if (e.scope === 'cross') {
              const subj = `f8.bus.${e.edgeId}`;
              queues.set(e.edgeId, []);
              const s = nc.subscribe(subj);
              (async () => {
                for await (const msg of s) {
                  const q = queues.get(e.edgeId);
                  if (!q) continue;
                  q.length = 0;
                  q.push(JSON.parse(sc.decode(msg.data)));
                  const max = e.queueSize || 64;
                  if (q.length > max) q.splice(0, q.length - max);
                }
              })();
            }
          }
        }
        log(`applied graph etag=${payload.etag}`);
      }
    })();

    // State fanout receive
    (async () => {
      const sub = nc.subscribe('f8.state.*.set');
      for await (const m of sub) {
        const parts = m.subject.split('.');
        const src = parts[2];
        if (src === id) continue;
        log(`received state from ${src}: ${sc.decode(m.data)}`);
      }
    })();

    // State snapshot request handler
    (async () => {
      const sub = nc.subscribe(`f8.state.${id}.snapshot`);
      for await (const m of sub) {
        const entry = await kvState.get(id).catch(() => null);
        const state = entry && entry.data ? JSON.parse(sc.decode(entry.data)) : null;
        m.respond(sc.encode(JSON.stringify({ state, etag: entry?.revision || 0 })));
      }
    })();

    // Tick loop
    setInterval(() => {
      for (const [edgeId, q] of queues.entries()) {
        if (q.length === 0) continue;
        const msg = q[q.length - 1];
        log(`consumed data on edge ${edgeId}: ${JSON.stringify(msg)}`);
        q.length = 0;
      }
      const state = { instanceId: id, ts: Date.now() };
      nc.publish(`f8.state.${id}.set`, sc.encode(JSON.stringify(state)));
      kvState.put(id, sc.encode(JSON.stringify(state))).catch(() => {});
    }, 300);
    return;
  }

  if (role === 'web') {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const request = async (subj, payload = {}, timeoutMs = 2000, attempts = 5, delayMs = 200) => {
      let lastErr;
      for (let i = 0; i < attempts; i++) {
        try {
          const m = await nc.request(subj, sc.encode(JSON.stringify(payload)), { timeout: timeoutMs });
          return JSON.parse(sc.decode(m.data));
        } catch (err) {
          lastErr = err;
          await sleep(delayMs);
        }
      }
      throw lastErr;
    };

    const publish = (subj, payload = {}) => nc.publish(subj, sc.encode(JSON.stringify(payload)));

    const graphV1 = { edges: [{ edgeId: 'edge1', scope: 'cross', strategy: 'latest', queueSize: 1, dropOld: true }] };
    const graphV2 = { edges: [{ edgeId: 'edge2', scope: 'cross', strategy: 'latest', queueSize: 1, dropOld: true }] };
    let etag = 0;

    await sleep(200);
    const snapshot = async () => request('f8.master.snapshot');

    log('ping master');
    log(JSON.stringify(await request('f8.master.ping')));

    const snap1 = await snapshot();
    etag = snap1.etag || 0;
    log('apply graph v1');
    const r1 = await request('f8.master.apply', { graph: graphV1, expectedEtag: etag });
    log(JSON.stringify(r1));
    if (r1.etag) etag = r1.etag;
    publish('f8.bus.edge1', { payload: { value: 1 }, ts: Date.now() });

    setTimeout(async () => {
      log('master down toggle');
      publish('f8.master.toggle', { up: false });
      log('apply graph v2 (expect read-only error)');
      log(JSON.stringify(await request('f8.master.apply', { graph: graphV2, expectedEtag: etag }).catch((e) => ({ error: e.message }))));
      publish('f8.bus.edge1', { payload: { value: 99 }, ts: Date.now() });
    }, 500);

    setTimeout(async () => {
      log('master up toggle');
      publish('f8.master.toggle', { up: true });
      const snap2 = await snapshot();
      etag = snap2.etag || etag;
      log('apply graph v2 (expect success)');
      const r2 = await request('f8.master.apply', { graph: graphV2, expectedEtag: etag });
      log(JSON.stringify(r2));
      if (r2.etag) etag = r2.etag;
      publish('f8.bus.edge2', { payload: { value: 7 }, ts: Date.now() });
      // fetch state snapshots after recovery
      try {
        const sA = await request('f8.state.engineA.snapshot', {}, 2000, 3, 200);
        const sB = await request('f8.state.engineB.snapshot', {}, 2000, 3, 200);
        log(`engineA snapshot: ${JSON.stringify(sA)}`);
        log(`engineB snapshot: ${JSON.stringify(sB)}`);
      } catch (e) {
        log(`snapshot fetch error: ${e.message}`);
      }
    }, 1000);

    setTimeout(async () => {
      log('demo done, exiting');
      try {
        await nc.close();
      } catch (_) {
        /* ignore */
      }
      process.exit(0);
    }, 1700);
  }
}
