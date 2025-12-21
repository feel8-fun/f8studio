# Web (TypeScript)

- Flow editor and control surface; runs in browser (Chrome/Edge).
- Connects to daemon via NATS bridge; falls back to web-only degraded mode when daemon absent.
- Generated clients from OpenAPI wrap NATS req/rep; add connection status and capability gating.
- Build: pnpm; keep codegen outputs under `packages/shared` or a subfolder to avoid drift.
