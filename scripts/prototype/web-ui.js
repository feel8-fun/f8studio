#!/usr/bin/env node
/**
 * Simple web UI to interact with the NATS prototype.
 * Provides endpoints to ping master, snapshot engine state, publish state, and publish data edges.
 *
 * Prereq: run nats-server -js and the prototype actors (run-nats.js) or an equivalent setup.
 *
 * Start: node scripts/prototype/web-ui.js
 * Open:  http://localhost:3000
 */

const express = require('express');
const path = require('path');
const { connect, StringCodec } = require('nats');

const PORT = process.env.PORT || 3001;
const NATS_URL = process.env.NATS_URL || 'nats://127.0.0.1:4222';

async function main() {
  const app = express();
  app.use(express.json());

  const sc = StringCodec();
  const nc = await connect({ servers: NATS_URL, name: 'web-ui' });
  const request = (subj, payload = {}, timeoutMs = 1500) =>
    nc.request(subj, sc.encode(JSON.stringify(payload)), { timeout: timeoutMs }).then((m) => JSON.parse(sc.decode(m.data)));
  const publish = (subj, payload = {}) => nc.publish(subj, sc.encode(JSON.stringify(payload)));

  app.get('/api/ping', async (_req, res) => {
    try {
      const resp = await request('f8.master.ping');
      res.json(resp);
    } catch (e) {
      res.status(500).json({ error: e.message || String(e) });
    }
  });

  app.get('/api/state/:id', async (req, res) => {
    try {
      const subj = `f8.state.${req.params.id}.snapshot`;
      const resp = await request(subj);
      res.json(resp);
    } catch (e) {
      res.status(500).json({ error: e.message || String(e) });
    }
  });

  app.post('/api/state/:id', async (req, res) => {
    try {
      const subj = `f8.state.${req.params.id}.set`;
      publish(subj, req.body || {});
      res.json({ ok: true, subject: subj });
    } catch (e) {
      res.status(500).json({ error: e.message || String(e) });
    }
  });

  app.post('/api/data/:edge', async (req, res) => {
    try {
      const subj = `f8.bus.${req.params.edge}`;
      publish(subj, req.body || {});
      res.json({ ok: true, subject: subj });
    } catch (e) {
      res.status(500).json({ error: e.message || String(e) });
    }
  });

  app.post('/api/master/toggle', async (req, res) => {
    try {
      publish('f8.master.toggle', { up: !!req.body.up });
      res.json({ ok: true, up: !!req.body.up });
    } catch (e) {
      res.status(500).json({ error: e.message || String(e) });
    }
  });

  app.use(express.static(path.join(__dirname, 'public')));

  app.listen(PORT, () => {
    console.log(`[web-ui] listening on http://localhost:${PORT} (NATS ${NATS_URL})`);
  });
}

main().catch((err) => {
  console.error('[web-ui] failed to start', err);
  process.exit(1);
});
