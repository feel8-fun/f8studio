/**
 * Minimal IPC bus client to mimic NATS-style pub/sub + KV over process.send.
 * Works with the orchestrator in run-multi.js.
 */
let counter = 0;
function uid() {
  counter += 1;
  return `${Date.now().toString(36)}-${Math.random().toString(16).slice(2)}-${counter}`;
}

const subs = new Map(); // subject -> handler[]
const pending = new Map(); // reqId -> {resolve,reject}

function onMessage(msg) {
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'pub') {
    const handlers = subs.get(msg.subject);
    if (handlers) handlers.forEach((h) => h(msg.data, msg.subject));
  } else if (msg.type === 'kv.resp') {
    const pendingReq = pending.get(msg.reqId);
    if (pendingReq) {
      pendingReq.resolve(msg);
      pending.delete(msg.reqId);
    }
  }
}

process.on('message', onMessage);

function publish(subject, data) {
  process.send?.({ type: 'pub', subject, data });
}

function subscribe(subject, handler) {
  if (!subs.has(subject)) subs.set(subject, []);
  subs.get(subject).push(handler);
  process.send?.({ type: 'sub', subject });
  return () => {
    const arr = subs.get(subject) || [];
    const idx = arr.indexOf(handler);
    if (idx >= 0) arr.splice(idx, 1);
    if (arr.length === 0) subs.delete(subject);
    process.send?.({ type: 'unsub', subject });
  };
}

function kvGet(bucket, key) {
  const reqId = uid();
  return new Promise((resolve) => {
    pending.set(reqId, { resolve });
    process.send?.({ type: 'kv.get', reqId, bucket, key });
  });
}

function kvPut(bucket, key, value, expected) {
  const reqId = uid();
  return new Promise((resolve) => {
    pending.set(reqId, { resolve });
    process.send?.({ type: 'kv.put', reqId, bucket, key, value, expected });
  });
}

function request(subject, payload) {
  const reply = `_inbox.${uid()}`;
  return new Promise((resolve) => {
    const unsub = subscribe(reply, (resp) => {
      unsub();
      resolve(resp);
    });
    publish(subject, { ...payload, reply });
  });
}

module.exports = {
  publish,
  subscribe,
  kvGet,
  kvPut,
  request,
};
