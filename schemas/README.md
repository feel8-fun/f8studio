# Schemas

- `common.schema.json`: Shared definitions (`stateField`, `port`) to avoid duplication.
- `service.schema.json`: Service profile (canonicalId, serviceSlug, label/description/tags, launch command, state model with access `rw|ro|init`, commands). `serviceSlug` is the safe token for NATS/KV; `canonicalId` can be reverse-DNS for registry.
- `operator.schema.json`: Operator spec (operatorType, label/description/tags, state model with access, ports, allowAdd flags) for validating runtime operator catalogs published by engine instances.
- Ports/state can optionally allow dynamic additions (`allowAddState`, `allowAddPorts`) to support plugin-driven nodes.
- Use these schemas for validation (C++: json-schema-validator; TS: ajv) and for codegen of shared types.
