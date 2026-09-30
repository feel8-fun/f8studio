# Schemas

`schemas/protocol.yml` (OpenAPI 3.0.3) defines the shared service and runtime
JSON models generated for Python and C++. Generated files are committed and
checked in CI; package builds do not generate them.

Other contracts have their own sources:

- `stream-wire.json`: binary video and audio frame headers.
- `runtime-keys.json`: runtime transport key templates.
- `runtime-control.json`: control endpoint names.
- `rungraph-fingerprint.json`: deployment fingerprint normalization.
- Studio document and HTTP contracts: typed models in `f8studio_core`,
  `f8studio_server`, and `f8media_protocol`; Web types are generated from them.

Key components (under `components.schemas`):
- `F8ServiceSpec`: Service profile (serviceClass, tags, launch, states, commands, ports, etc.)
- `F8ServiceEntry`: Discovery entry stored in `services/**/service.yml`
- `F8OperatorSpec`: Operator spec for runtime catalogs published by engine instances
- `F8Edge`: Edge record (`kind`, strategy/queue/timeout for cross-service data edges)
- `F8DataTypeSchema`: Value schema used by ports/state/params
