import express from "express";
import swaggerUi from "swagger-ui-express";
import { readFileSync } from "fs";
import { resolve } from "path";
import { randomUUID } from "crypto";
import { connect, ErrorCode, StringCodec } from "nats";
import { parse as parseYaml } from "yaml";

const PORT = Number(process.env.PORT ?? 8080);
const NATS_URL = process.env.NATS_URL ?? "nats://localhost:4222";
const OPENAPI_PATH = resolve(process.env.OPENAPI_PATH ?? "api/specs/master.yaml");
const DEFAULT_CLIENT_ID = process.env.CLIENT_ID ?? "http-gateway";
const API_VERSION = process.env.API_VERSION ?? "v1";

const sc = StringCodec();
const ENVELOPE_KEYS = new Set([
  "msgId",
  "traceId",
  "clientId",
  "hop",
  "ts",
  "apiVersion",
  "payload",
  "headers",
]);

function loadOpenApiSpec(path) {
  const raw = readFileSync(path, "utf8");
  if (path.endsWith(".yaml") || path.endsWith(".yml")) {
    return parseYaml(raw);
  }
  return JSON.parse(raw);
}

function isPlainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function extractPayload(body) {
  if (!isPlainObject(body)) {
    return body ?? {};
  }
  if (Object.prototype.hasOwnProperty.call(body, "payload")) {
    return body.payload ?? {};
  }
  const payload = {};
  for (const [key, value] of Object.entries(body)) {
    if (!ENVELOPE_KEYS.has(key)) {
      payload[key] = value;
    }
  }
  return payload;
}

function normalizeEnvelope(body, req) {
  const envelope = isPlainObject(body) ? body : {};
  const now = new Date().toISOString();
  return {
    msgId: envelope.msgId ?? randomUUID(),
    traceId: envelope.traceId ?? req.header("x-trace-id") ?? randomUUID(),
    clientId: envelope.clientId ?? req.header("x-client-id") ?? DEFAULT_CLIENT_ID,
    hop: envelope.hop ?? 0,
    ts: envelope.ts ?? now,
    apiVersion: envelope.apiVersion ?? API_VERSION,
    headers: envelope.headers ?? {},
    payload: extractPayload(envelope),
  };
}

function decodeReply(data) {
  const text = sc.decode(data);
  return JSON.parse(text);
}

function wrapErrorEnvelope(template, errorCode, errorMessage) {
  return {
    ...template,
    payload: {
      status: "error",
      errorCode,
      errorMessage,
    },
  };
}

function mapNatsError(err, envelope) {
  if (err?.code === ErrorCode.Timeout) {
    return { statusCode: 504, body: wrapErrorEnvelope(envelope, "timeout", "NATS request timed out") };
  }
  if (err?.code === ErrorCode.NoResponders) {
    return {
      statusCode: 503,
      body: wrapErrorEnvelope(envelope, "unavailable", "No NATS responders for subject"),
    };
  }
  return {
    statusCode: 502,
    body: wrapErrorEnvelope(envelope, "internal", err?.message ?? "NATS request failed"),
  };
}

function createHandler(nc, subject, timeoutMs) {
  return async (req, res) => {
    const envelope = normalizeEnvelope(req.body ?? {}, req);
    try {
      const msg = await nc.request(subject, sc.encode(JSON.stringify(envelope)), { timeout: timeoutMs });
      const reply = decodeReply(msg.data);
      res.status(200).json(reply);
    } catch (err) {
      const { statusCode, body } = mapNatsError(err, envelope);
      res.status(statusCode).json(body);
    }
  };
}

async function main() {
  const spec = loadOpenApiSpec(OPENAPI_PATH);
  const nc = await connect({ servers: NATS_URL });

  const app = express();
  app.use(express.json({ limit: "1mb" }));

  // Serve external schema references so Swagger can resolve $ref links.
  app.use("/schemas", express.static(resolve("schemas")));

  app.get("/healthz", (_req, res) => res.json({ status: "ok" }));
  app.get("/openapi.json", (_req, res) => res.json(spec));
  app.use("/docs", swaggerUi.serve, swaggerUi.setup(spec));
  app.get("/", (_req, res) => res.redirect("/docs"));

  app.post("/ping", createHandler(nc, "f8.master.ping", 1000));
  app.post("/config/apply", createHandler(nc, "f8.master.config.apply", 3000));
  app.post("/registry/register", createHandler(nc, "f8.master.registry.register", 2000));
  app.post("/registry/unregister", createHandler(nc, "f8.master.registry.unregister", 2000));

  app.use((req, res) => res.status(404).json({ status: "error", errorCode: "not-found", path: req.path }));

  app.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`HTTP gateway listening on http://localhost:${PORT} -> ${NATS_URL}`);
  });

  nc.closed().then((err) => {
    if (err) {
      // eslint-disable-next-line no-console
      console.error("NATS connection closed with error", err);
      process.exit(1);
    }
  });
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error("Failed to start HTTP gateway", err);
  process.exit(1);
});
