# Using OpenAPI specs with NATS

- Primary transport is NATS request/reply for master/engine/other services. HTTP is optional via a single gateway that proxies HTTP→NATS (e.g., `scripts/http-gateway.mjs`) and serves Swagger UI.
- Specs define RPC contracts (not HTTP). Keep `operationId`, `x-nats-subject`, `x-nats-timeout-ms`.
- Run codegen to get models/types (TS/C++) from `api/specs/*.yaml`.
- Wrap generated types with thin NATS adapters (client + handler) that map `operationId` to `x-nats-subject`.
- Use schema-based tests (contract tests) to validate payloads.

Example codegen commands (adjust paths/tools):
```bash
# TS models only
openapi-generator-cli generate -i api/specs/master.yaml -g typescript-fetch -o packages/shared/generated/master --additional-properties=modelPropertyNaming=original

# C++ models only
openapi-generator-cli generate -i api/specs/master.yaml -g cpp-pistache-server -o packages/shared/generated-cpp/master
```
Use only the models/types; ignore generated HTTP transport.

Example adapter pattern is in:
- `packages/shared/nats-adapter.ts` (client-side TS)
- `packages/daemon/nats_adapter_example.cpp` (server-side C++ sketch)
