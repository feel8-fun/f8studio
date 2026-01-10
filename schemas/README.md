# Schemas

Single source of truth: `schemas/protocol.yml` (OpenAPI 3.0.3).

Key components (under `components.schemas`):
- `F8ServiceSpec`: Service profile (serviceClass, tags, launch, states, commands, ports, etc.)
- `F8ServiceEntry`: Discovery entry stored in `services/**/service.yml`
- `F8OperatorSpec`: Operator spec for runtime catalogs published by engine instances
- `F8Edge`: Edge record (`kind`, strategy/queue/timeout for cross-service data edges)
- `F8DataTypeSchema`: Value schema used by ports/state/params
