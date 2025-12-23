/**
 * Minimal in-memory prototype for f8 master/engine/web interaction.
 * Focus: state/data subjects with f8.* prefix, etag/readonly on master loss,
 * and cross-instance data queue with "latest + drop-old".
 *
 * Run: node scripts/prototype/sim.js
 */

class Bus {
  constructor() {
    this.subs = new Map(); // subject -> Set<fn>
  }
  subscribe(subject, fn) {
    if (!this.subs.has(subject)) this.subs.set(subject, new Set());
    this.subs.get(subject).add(fn);
    return () => this.subs.get(subject)?.delete(fn);
  }
  publish(subject, payload) {
    const subs = this.subs.get(subject);
    if (!subs || subs.size === 0) return;
    for (const fn of Array.from(subs)) {
      fn(payload, subject);
    }
  }
}

class Master {
  constructor(bus) {
    this.bus = bus;
    this.up = true;
    this.etag = 0;
    this.graph = null;
  }
  ping() {
    return this.up ? { status: 'ok', etag: this.etag } : { status: 'unavailable' };
  }
  snapshot() {
    if (!this.up) return { error: 'MASTER_UNAVAILABLE' };
    return { graph: this.graph, etag: this.etag };
  }
  applyGraph(graphBlob, expectedEtag) {
    if (!this.up) return { error: 'MASTER_UNAVAILABLE', message: 'read-only; master down' };
    if (expectedEtag !== this.etag) {
      return { error: 'ETAG_MISMATCH', currentEtag: this.etag };
    }
    this.etag += 1;
    this.graph = graphBlob;
    this.bus.publish('f8.control.apply', { graph: graphBlob, etag: this.etag });
    return { ok: true, etag: this.etag };
  }
}

class Engine {
  constructor(id, bus) {
    this.id = id;
    this.bus = bus;
    this.graph = null;
    this.graphEtag = null;
    this.state = {};
    this.remoteState = {};
    this.dataQueues = new Map(); // edgeId -> array

    // Listen for apply commands targeted to all engines.
    this.bus.subscribe('f8.control.apply', (payload) => {
      this.graph = payload.graph;
      this.graphEtag = payload.etag;
      this._installSubscriptions();
      log(`engine ${this.id}: applied graph etag=${payload.etag}`);
    });

    // Listen to state fanout from others.
    this.bus.subscribe('f8.state.broadcast', (payload) => {
      if (payload.instanceId === this.id) return;
      this.remoteState[payload.instanceId] = payload.state;
      log(`engine ${this.id}: received state from ${payload.instanceId}: ${JSON.stringify(payload.state)}`);
    });
  }

  _installSubscriptions() {
    // Clear old data queue subs
    this.dataQueues.clear();
    if (!this.graph || !Array.isArray(this.graph.edges)) return;
    for (const edge of this.graph.edges) {
      if (edge.scope === 'cross') {
        const subject = `f8.bus.${edge.edgeId}`;
        this.dataQueues.set(edge.edgeId, []);
        this.bus.subscribe(subject, (msg) => {
          const q = this.dataQueues.get(edge.edgeId);
          if (!q) return;
          // strategy: latest + drop-old
          q.length = 0;
          q.push(msg);
          if (q.length > (edge.queueSize || 64)) q.shift();
        });
      }
    }
  }

  tick() {
    // Consume data queues (latest)
    for (const [edgeId, q] of this.dataQueues.entries()) {
      if (q.length === 0) continue;
      const msg = q[q.length - 1];
      log(`engine ${this.id}: consumed data on edge ${edgeId}: ${JSON.stringify(msg)}`);
      q.length = 0;
    }
    // Fanout local state
    this.bus.publish('f8.state.broadcast', { instanceId: this.id, state: this.state });
  }
}

function log(msg) {
  const now = new Date().toISOString();
  console.log(`[${now}] ${msg}`);
}

function runDemo() {
  const bus = new Bus();
  const master = new Master(bus);
  const e1 = new Engine('engineA', bus);
  const e2 = new Engine('engineB', bus);

  // Client: apply initial graph
  const graphV1 = {
    edges: [
      { edgeId: 'edge1', scope: 'cross', strategy: 'latest', queueSize: 64, dropOld: true },
    ],
  };
  log('client: apply graph v1');
  log(JSON.stringify(master.applyGraph(graphV1, 0)));

  // Publish cross-instance data and tick
  bus.publish('f8.bus.edge1', { payload: { value: 42 }, ts: Date.now() });
  e1.state = { foo: 1 };
  e2.state = { bar: 2 };
  e1.tick();
  e2.tick();

  // Simulate master outage
  master.up = false;
  log('client: master down, attempt apply graph v2 (expect read-only)');
  const graphV2 = { edges: [{ edgeId: 'edge2', scope: 'cross', strategy: 'latest', queueSize: 64, dropOld: true }] };
  log(JSON.stringify(master.applyGraph(graphV2, master.etag)));

  // Engines keep ticking on old graph
  bus.publish('f8.bus.edge1', { payload: { value: 99 }, ts: Date.now() });
  e1.tick();
  e2.tick();

  // Master recovers and apply succeeds
  master.up = true;
  log('client: master recovered, apply graph v2');
  log(JSON.stringify(master.applyGraph(graphV2, master.etag)));

  // New data on edge2 (only present in v2)
  bus.publish('f8.bus.edge2', { payload: { value: 7 }, ts: Date.now() });
  e1.tick();
  e2.tick();
}

if (require.main === module) {
  runDemo();
}
