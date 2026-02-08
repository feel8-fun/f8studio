/**
 * Lovense Local API mock/capture server for this game.
 *
 * The game (mobile mode) POSTs JSON to:
 *   http://<ip>:<port>/command
 *
 * This server responds to GetToys and captures Pattern/Function commands
 * (vibration + Solace thrusting) plus Stop.
 *
 * Usage (PowerShell):
 *   node .\\lovense-mock-server.js
 *   # or:
 *   $env:LOVENSE_PORT=30010; node .\\lovense-mock-server.js
 *
 * Output:
 *   - logs to console
 *   - appends NDJSON to ./lovense-capture.ndjson
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const HOST = process.env.LOVENSE_HOST || "0.0.0.0";
const PORT = Number(process.env.LOVENSE_PORT || "30010");
const CAPTURE_PATH =
  process.env.LOVENSE_CAPTURE_PATH ||
  path.resolve(process.cwd(), "lovense-capture.ndjson");
const PRINT_RAW = String(process.env.LOVENSE_PRINT_RAW || "").trim() === "1";
const PRINT_PRETTY = String(process.env.LOVENSE_PRINT_PRETTY || "").trim() === "1";
const PRINT_HEADERS = String(process.env.LOVENSE_PRINT_HEADERS || "").trim() === "1";
const PRINT_RESP = String(process.env.LOVENSE_PRINT_RESP || "").trim() === "1";

function nowIso() {
  return new Date().toISOString();
}

function writeJson(res, statusCode, obj) {
  const body = JSON.stringify(obj);
  if (PRINT_RESP) {
    const ts = nowIso();
    try {
      console.log(`[${ts}] resp status=${statusCode} body=${body}`);
    } catch {
      // ignore
    }
  }
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-cache, no-store, max-age=0",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept",
  });
  res.end(body);
}

function readUtf8OrHex(buf) {
  try {
    return buf.toString("utf8");
  } catch {
    return buf.toString("hex");
  }
}

function appendCapture(entry) {
  try {
    fs.appendFileSync(CAPTURE_PATH, JSON.stringify(entry) + "\n", "utf8");
  } catch (e) {
    // Best-effort capture; still keep the server running.
    console.error(`[${nowIso()}] capture write failed:`, e && e.message);
  }
}

function printHeadersLine(ts, headers, contentType) {
  if (!PRINT_HEADERS) return;
  const h = headers || {};
  const picked = {
    "content-type": h["content-type"],
    "content-length": h["content-length"],
    "user-agent": h["user-agent"],
    "accept": h["accept"],
    "accept-encoding": h["accept-encoding"],
    "connection": h["connection"],
  };
  console.log(`[${ts}] headers=${JSON.stringify(picked)}`);
  if (contentType && !picked["content-type"]) {
    console.log(`[${ts}] content-type=${String(contentType)}`);
  }
}

function wsAcceptKey(secWebSocketKey) {
  // https://datatracker.ietf.org/doc/html/rfc6455#section-1.3
  return crypto
    .createHash("sha1")
    .update(secWebSocketKey + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11", "utf8")
    .digest("base64");
}

function wsSendText(socket, text) {
  const payload = Buffer.from(String(text), "utf8");
  const header = [];
  header.push(0x81); // FIN + text
  if (payload.length < 126) {
    header.push(payload.length);
  } else if (payload.length < 65536) {
    header.push(126, (payload.length >> 8) & 0xff, payload.length & 0xff);
  } else {
    // 64-bit length (we won't hit this, but keep it correct)
    header.push(127, 0, 0, 0, 0);
    header.push((payload.length >> 24) & 0xff);
    header.push((payload.length >> 16) & 0xff);
    header.push((payload.length >> 8) & 0xff);
    header.push(payload.length & 0xff);
  }
  socket.write(Buffer.concat([Buffer.from(header), payload]));
}

function wsSendPong(socket, payload) {
  const p = payload ? Buffer.from(payload) : Buffer.alloc(0);
  const header = [];
  header.push(0x8a); // FIN + pong
  header.push(p.length);
  socket.write(Buffer.concat([Buffer.from(header), p]));
}

function wsSendClose(socket) {
  socket.write(Buffer.from([0x88, 0x00])); // FIN + close, no payload
  socket.end();
}

function wsTryParseFrames(state, chunk) {
  state.buffer = Buffer.concat([state.buffer, chunk]);
  const messages = [];

  while (state.buffer.length >= 2) {
    const b0 = state.buffer[0];
    const b1 = state.buffer[1];
    const fin = (b0 & 0x80) !== 0;
    const opcode = b0 & 0x0f;
    const masked = (b1 & 0x80) !== 0;
    let len = b1 & 0x7f;
    let offset = 2;

    if (len === 126) {
      if (state.buffer.length < offset + 2) break;
      len = state.buffer.readUInt16BE(offset);
      offset += 2;
    } else if (len === 127) {
      if (state.buffer.length < offset + 8) break;
      // Only support up to 32-bit lengths here (good enough for JSON control messages)
      const hi = state.buffer.readUInt32BE(offset);
      const lo = state.buffer.readUInt32BE(offset + 4);
      if (hi !== 0) throw new Error("WS frame too large");
      len = lo;
      offset += 8;
    }

    const maskLen = masked ? 4 : 0;
    if (state.buffer.length < offset + maskLen + len) break;

    let maskKey = null;
    if (masked) {
      maskKey = state.buffer.subarray(offset, offset + 4);
      offset += 4;
    }
    let payload = state.buffer.subarray(offset, offset + len);
    state.buffer = state.buffer.subarray(offset + len);

    if (masked && maskKey) {
      const unmasked = Buffer.alloc(payload.length);
      for (let i = 0; i < payload.length; i++) {
        unmasked[i] = payload[i] ^ maskKey[i % 4];
      }
      payload = unmasked;
    }

    if (!fin) {
      // Fragmentation not expected here; ignore
      continue;
    }

    messages.push({ opcode, payload });
  }

  return messages;
}

function printCommandLine(ts, summary, raw) {
  if (!summary || typeof summary !== "object") {
    console.log(`[${ts}] unknown`, raw);
    return;
  }

  switch (summary.type) {
    case "get_toys":
      console.log(`[${ts}] GetToys apiVer=${summary.apiVer ?? ""}`.trim());
      break;
    case "ping":
      console.log(`[${ts}] ping`);
      break;
    case "pong":
      console.log(`[${ts}] pong`);
      break;
    case "vibration_pattern":
      console.log(
        `[${ts}] Pattern toy=${summary.toy} timeSec=${summary.timeSec} strength=${summary.strength} apiVer=${summary.apiVer}`
      );
      break;
    case "solace_thrusting":
      console.log(
        `[${ts}] Function(Thrusting) toy=${summary.toy} thrusting=${summary.thrusting} depth=${summary.depth} timeSec=${summary.timeSec} loopRunningSec=${summary.loopRunningSec ?? ""} loopPauseSec=${summary.loopPauseSec ?? ""} apiVer=${summary.apiVer}`
      );
      break;
    case "all_vibrate":
      console.log(
        `[${ts}] Function(All) level=${summary.all} timeSec=${summary.timeSec} apiVer=${summary.apiVer}`
      );
      break;
    case "stop":
      console.log(
        `[${ts}] Function(Stop) toy=${summary.toy} timeSec=${summary.timeSec} apiVer=${summary.apiVer}`
      );
      break;
    default:
      console.log(`[${ts}] ${summary.type}`, summary);
      break;
  }

  if (PRINT_RAW) {
    if (PRINT_PRETTY) console.log(`[${ts}] raw=`, JSON.stringify(raw, null, 2));
    else console.log(`[${ts}] raw=${JSON.stringify(raw)}`);
  }
}

function parseBody(bodyText, contentType) {
  const trimmed = String(bodyText || "").trim();
  const ct = String(contentType || "").toLowerCase();

  // Try JSON if it looks like JSON, regardless of content-type.
  if (
    ct.includes("application/json") ||
    trimmed.startsWith("{") ||
    trimmed.startsWith("[") ||
    trimmed.startsWith('"')
  ) {
    return JSON.parse(trimmed);
  }

  // Handle x-www-form-urlencoded.
  if (ct.includes("application/x-www-form-urlencoded")) {
    const params = new URLSearchParams(trimmed);
    const obj = {};
    for (const [k, v] of params.entries()) obj[k] = v;
    return obj;
  }

  // Fallback: raw text.
  return trimmed;
}

function normalizePayload(value) {
  // Some clients send JSON-string payloads, or nest JSON under `data`.
  let current = value;

  for (let i = 0; i < 3; i++) {
    if (typeof current === "string") {
      const s = current.trim();
      if (
        (s.startsWith("{") && s.endsWith("}")) ||
        (s.startsWith("[") && s.endsWith("]")) ||
        (s.startsWith('"') && s.endsWith('"'))
      ) {
        try {
          current = JSON.parse(s);
          continue;
        } catch {
          return { value: current };
        }
      }
      return { value: current };
    }

    if (Array.isArray(current)) {
      // Common pattern: single-element array wrapping an object.
      if (current.length === 1 && current[0] && typeof current[0] === "object") {
        current = current[0];
        continue;
      }
      return { array: current };
    }

    if (current && typeof current === "object") {
      // If nested JSON is present in `data`, unwrap it.
      if (typeof current.data === "string") {
        const s = current.data.trim();
        if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
          try {
            current = JSON.parse(s);
            continue;
          } catch {
            // ignore
          }
        }
      }
      return current;
    }
  }

  return { value: current };
}

// Minimal toy list so the game considers itself "connected".
// - "lush" will use Pattern (vibration)
// - "solace" will use Function Thrusting/Depth
function buildToyMap() {
  // Object.values(toyMap) is used in-game; keep numeric-ish keys for stable order.
  return {
    0: {
      nickName: "Mock Lush",
      name: "lush",
      id: "MOCK_LUSH_0",
      battery: 100,
      // Some SDKs use fVersion/hVersion, some use version.
      fVersion: 0,
      hVersion: 0,
      version: "3",
      connected: true,
      status: "1",
      domain: "127.0.0.1",
      port: PORT,
      isHttps: false,
      platform: "mock",
    },
    1: {
      nickName: "Mock Solace",
      name: "solace",
      id: "MOCK_SOLACE_1",
      battery: 100,
      fVersion: 0,
      hVersion: 0,
      version: "1",
      connected: true,
      status: "1",
      domain: "127.0.0.1",
      port: PORT,
      isHttps: false,
      platform: "mock",
    },
  };
}

function buildGetToysResponse(toyMap) {
  // Unity SDK in LovenseRemote.dll appears to model:
  // - AllLovenseToysResult { code:int, type:string, data:AllLovenseToys }
  // - AllLovenseToys { toys:string, allToys:dict, appType, platform, gameAppId }
  //
  // RPG Maker implementations often model:
  // - { data: { toys: "<json string>" } }
  const toysString = JSON.stringify(toyMap);
  const allToysById = {};
  for (const toy of Object.values(toyMap)) {
    if (toy && toy.id) allToysById[toy.id] = toy;
  }

  return {
    // Unity-friendly
    code: 0,
    type: "GetToys",
    message: "OK",
    data: {
      toys: toysString,
      allToys: allToysById,
      appType: "remote",
      platform: "pc",
      gameAppId: "",
    },

    // RPGM-friendly
    ok: true,
    data2: {
      toys: toysString,
      toysMap: toyMap,
    },

    // Convenience/debug
    toys: toyMap,
  };
}

function summarizeCommand(payload) {
  if (!payload || typeof payload !== "object") return { type: "unknown" };

  const cmd =
    payload.command ??
    payload.cmd ??
    payload.type ?? // some clients use `type` instead of `command`
    payload.request ??
    payload.method;
  if (cmd === "ping") return { type: "ping" };
  if (cmd === "pong") return { type: "pong" };
  if (cmd === "Pattern") {
    return {
      type: "vibration_pattern",
      toy: payload.toy,
      timeSec: payload.timeSec,
      strength: payload.strength,
      rule: payload.rule,
      apiVer: payload.apiVer,
    };
  }

  if (cmd === "Function") {
    const action = String(payload.action || "");
    const m = action.match(/^Thrusting:(\d+),Depth:(\d+)$/);
    const all = action.match(/^All:(\d+)$/);
    return {
      type: m
        ? "solace_thrusting"
        : all
          ? "all_vibrate"
          : action === "Stop"
            ? "stop"
            : "function",
      toy: payload.toy,
      timeSec: payload.timeSec,
      action,
      thrusting: m ? Number(m[1]) : undefined,
      depth: m ? Number(m[2]) : undefined,
      all: all ? Number(all[1]) : undefined,
      loopRunningSec: payload.loopRunningSec,
      loopPauseSec: payload.loopPauseSec,
      apiVer: payload.apiVer,
    };
  }

  if (cmd === "GetToys") return { type: "get_toys", apiVer: payload.apiVer ?? 1 };
  if (cmd === "PatternV2") return { type: "pattern_v2", apiVer: payload.apiVer };

  return {
    type: "other",
    command: cmd,
    apiVer: payload.apiVer,
    keys: Object.keys(payload).slice(0, 20),
  };
}

const server = http.createServer((req, res) => {
  // CORS preflight (safe even if not needed in NW.js)
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Accept",
      "Access-Control-Max-Age": "86400",
    });
    return res.end();
  }

  // Log unexpected endpoints (Unity SDK may also attempt WS at /v1, which comes via 'upgrade',
  // but some clients may probe via plain HTTP too).
  if (req.url !== "/command") {
    const ts = nowIso();
    printHeadersLine(ts, req.headers, req.headers["content-type"] || "");
    if (PRINT_RAW) console.log(`[${ts}] http ${req.method} ${req.url}`);
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    return res.end("Not Found");
  }
  if (req.method !== "POST") {
    res.writeHead(405, { "Content-Type": "text/plain; charset=utf-8" });
    return res.end("Method Not Allowed");
  }

  let body = "";
  req.setEncoding("utf8");
  req.on("data", (chunk) => {
    body += chunk;
    // Prevent unbounded growth; Lovense payloads are tiny.
    if (body.length > 1024 * 1024) req.destroy();
  });
  req.on("end", () => {
    const contentType = req.headers["content-type"] || "";
    let rawPayload;
    try {
      rawPayload = parseBody(body, contentType);
    } catch (e) {
      const entry = {
        ts: nowIso(),
        error: "parse_failed",
        message: e && e.message,
        contentType,
        bodyText: body,
      };
      appendCapture(entry);
      console.error(`[${entry.ts}] parse failed:`, entry.message);
      return writeJson(res, 400, { ok: false, error: "parse_failed" });
    }

    const normalizedPayload = normalizePayload(rawPayload);

    const entry = {
      ts: nowIso(),
      remote: req.socket.remoteAddress,
      path: req.url,
      method: req.method,
      contentType,
      headers: PRINT_HEADERS ? req.headers : undefined,
      bodyText: body,
      raw: rawPayload,
      normalized: normalizedPayload,
      summary: summarizeCommand(normalizedPayload),
    };
    appendCapture(entry);

    if (entry.summary.type === "ping") {
      // Some Unity integrations keep-alive with {type:"ping"} before/while connecting.
      printHeadersLine(entry.ts, req.headers, contentType);
      printCommandLine(entry.ts, entry.summary, entry.raw);
      return writeJson(res, 200, { type: "pong" });
    }

    if (entry.summary.type === "get_toys") {
      const toyMap = buildToyMap();
      printHeadersLine(entry.ts, req.headers, contentType);
      printCommandLine(entry.ts, entry.summary, entry.raw);
      return writeJson(res, 200, buildGetToysResponse(toyMap));
    }

    // For all other commands, the game doesn't require any specific response.
    printHeadersLine(entry.ts, req.headers, contentType);
    printCommandLine(entry.ts, entry.summary, entry.raw);
    return writeJson(res, 200, { ok: true });
  });
});

server.on("upgrade", (req, socket, head) => {
  const ts = nowIso();
  const url = req.url || "";
  if (PRINT_HEADERS) console.log(`[${ts}] WS upgrade ${url} headers=${JSON.stringify(req.headers)}`);

  if (url !== "/v1") {
    socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
    socket.destroy();
    return;
  }

  const key = req.headers["sec-websocket-key"];
  if (!key) {
    socket.write("HTTP/1.1 400 Bad Request\r\n\r\n");
    socket.destroy();
    return;
  }

  const accept = wsAcceptKey(String(key));
  socket.write(
    [
      "HTTP/1.1 101 Switching Protocols",
      "Upgrade: websocket",
      "Connection: Upgrade",
      `Sec-WebSocket-Accept: ${accept}`,
      "\r\n",
    ].join("\r\n")
  );

  const state = { buffer: Buffer.alloc(0) };
  if (head && head.length) state.buffer = Buffer.from(head);

  console.log(`[${ts}] WS connected on /v1`);

  // Send a conservative initial sequence; client may ignore what it doesn't understand.
  wsSendText(socket, JSON.stringify({ type: "access-granted" }));
  wsSendText(socket, JSON.stringify({ type: "toy-list", data: { toys: buildToyMap() } }));

  socket.on("data", (chunk) => {
    let frames = [];
    try {
      frames = wsTryParseFrames(state, chunk);
    } catch (e) {
      console.error(`[${nowIso()}] WS frame parse error:`, e && e.message);
      wsSendClose(socket);
      return;
    }

    for (const f of frames) {
      if (f.opcode === 0x8) {
        wsSendClose(socket);
        return;
      }
      if (f.opcode === 0x9) {
        wsSendPong(socket, f.payload);
        continue;
      }
      if (f.opcode !== 0x1) continue;

      const text = readUtf8OrHex(f.payload);
      const ets = nowIso();
      console.log(`[${ets}] WS recv: ${text}`);

      // Best-effort responses for common client messages.
      try {
        const msg = JSON.parse(text);
        const t = msg.type || msg.command || msg.cmd;
        if (t === "ping") wsSendText(socket, JSON.stringify({ type: "pong" }));
        if (t === "access") wsSendText(socket, JSON.stringify({ type: "access-granted" }));
      } catch {
        // ignore
      }
    }
  });

  socket.on("error", (e) => {
    console.error(`[${nowIso()}] WS socket error:`, e && e.message);
  });
});

server.on("error", (err) => {
  const ts = nowIso();
  if (err && err.code === "EADDRINUSE") {
    console.error(`[${ts}] Port already in use: ${HOST}:${PORT}`);
    console.error(
      `[${ts}] Tip: pick another port via LOVENSE_PORT, or stop the process currently using it.`
    );
    process.exitCode = 1;
    return;
  }
  console.error(`[${ts}] Server error:`, err);
  process.exitCode = 1;
});

server.listen(PORT, HOST, () => {
  console.log(`[${nowIso()}] Lovense mock listening on http://${HOST}:${PORT}/command`);
  console.log(`[${nowIso()}] Capturing to ${CAPTURE_PATH}`);
  console.log(
    `[${nowIso()}] Print raw JSON: set LOVENSE_PRINT_RAW=1 (optional: LOVENSE_PRINT_PRETTY=1)`
  );
  console.log(
    `[${nowIso()}] Print request headers: set LOVENSE_PRINT_HEADERS=1`
  );
  const connectHost = HOST === "0.0.0.0" ? "127.0.0.1" : HOST;
  console.log(
    `[${nowIso()}] In-game: set Lovense connection type = Mobile, IP=${connectHost}, Port=${PORT}, then "Connect Toys".`
  );
});
