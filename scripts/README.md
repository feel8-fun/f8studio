# Scripts

- `codegen-*`: generate TS/C++ stubs from OpenAPI (NATS binding aware).
- `lint-*`: schema/spec/profile validation.
- `nats-dev`: start local in-memory NATS for tests/prototype.
- `http-gateway`: lightweight Express proxy that forwards HTTP calls to the NATS API and serves Swagger UI from `api/specs/master.yaml`.
- `validate-operator`: validate `services/*/operators.json` against `schemas/operator.schema.json`.
- `validate-service`: validate `services/*/service.json` against `schemas/service.schema.json`.
- `validate-edge`: validate edge lists against `schemas/edge.schema.json`.

TBD: add actual scripts once tool choices are finalized.
