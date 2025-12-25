# Schemas

- `common.schema.json`: Shared definitions (`stateField`, `port`) to avoid duplication.
- `service.schema.json`: Service profile (serviceClass, label/description/tags, launch command, state model with access `rw|ro|init`, commands). `serviceClass` should be reverse-DNS for registry (e.g., `fun.feel8.engine`).
- `operator.schema.json`: Operator spec (operatorClass, label/description/tags, state model with access, data ports `dataInPorts`/`dataOutPorts`, exec ports `execInPorts`/`execOutPorts`, allowAdd flags) for validating runtime operator catalogs published by engine instances; prefer `fun.feel8.op.*` prefixes to distinguish operators.
- Ports/state can optionally allow dynamic additions (`allowAddState`, data/exec allowAdd flags) to support plugin-driven nodes.
- Use these schemas for validation (C++: json-schema-validator; TS: ajv) and for codegen of shared types.
