# f8studio (greenfield skeleton)

Fresh workspace for the API-first, NATS-only architecture. Current focus is contracts, profiles, flows, and prototype scaffolding; no legacy code carried over.

## Layout
- `api/specs` — OpenAPI contracts (used for codegen) with NATS bindings via `x-nats-*`.
- `api/bindings` — shared envelope/error models and binding notes.
- `profiles` — platform/feature profile schemas and examples.
- `docs/flows` — sequence/state docs for connection, config, playback, degrade/recover.
- `packages/daemon` — C++ daemon (libmpv + state manager + NATS microservices).
- `packages/web` — TS client (web-only + enhanced via daemon), flow editor hooks.
- `packages/shared` — shared models/types; codegen outputs wrappers.
- `tests/contract` — spec-driven contract tests (NATS in-memory).
- `tests/integration` — end-to-end scenarios (web↔daemon).
- `scripts` — codegen, lint, local NATS bootstrap helpers.

## Next steps
- Hook `api/master.yaml` and `schemas/protocol.yml` into codegen/validation flow.
- Add scripts to generate TS/C++ stubs from OpenAPI and wrap NATS req/rep.
- Wire a minimal prototype: Web ping/echo + config apply via NATS in-memory broker.
