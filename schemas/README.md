# Studio schemas

`studio-api.gen.json` and `extensions.gen.json` are generated from the typed
Studio/SDK models by `scripts/web_studio/generate_contracts.py`.

Canonical shared service and runtime contracts live in the independent SDK
repository at `sdk/schemas/`. Edit those sources and run `pixi run protocol_codegen_all`
to update SDK models and their Studio TypeScript consumers.
