# Web Studio Architecture

Feel8 Studio is a local Web application. The browser owns graph interaction and presentation. Typed Python services own persistence, compilation, deployment, device integration, automation, and model credentials. Real-time media conversion and WebRTC peers run in a separate gateway process.

## Packages

| Package | Responsibility |
| --- | --- |
| `f8studio_core` | Graph document, patch rules, catalog contracts, validation, deterministic compilation |
| `f8studio_server` | HTTP/WebSocket application, SQLite repositories, runtime jobs, local integration, Agent/CLI/MCP boundary |
| `f8media_protocol` | Typed media gateway control contract and client |
| `f8media_gateway` | Zenoh subscriptions, frame/audio conversion, WebRTC peer lifecycle |
| `f8studio_web` | React graph editor, Inspector, assets, Agent workspace, Monaco and presentation renderers |

Dependencies point inward: core has no server, browser, media implementation, or GUI dependency. The server consumes core and the media protocol. The media gateway remains independently replaceable by a native implementation while preserving `f8media-api/1`.

## Authoritative State

`StudioApplication` is the single application service used by browser routes, Agent tools, CLI, and MCP. Project graph and layout revisions are committed transactionally to SQLite. Reliable graph/job events carry sequence and server epoch information; presentation and monitor events use bounded best-effort queues.

The browser projects the authoritative document into React Flow. Local drag and connection gestures produce typed patch operations. It does not compile runtime graphs or write service state directly outside the server contract.

## Runtime and Media

Deployment compiles a pure `StudioDocument` into runtime graphs. Service processes communicate through Zenoh. High-frequency FPS, latency, dropped-frame and processing counters remain monitor/data telemetry and are never modeled as service state fields.

The media gateway is a separate process so video decode/convert/encode work cannot block the Studio control plane. The server proxies signaling requests through the typed protocol. Browser ICE configuration comes from `/api/media/rtc-configuration`; strict SSH/VPN deployments can force TURN relay.

## Extension Model

There is no dynamic GUI plugin loader. Repository-owned capabilities register explicitly in typed backend registries and frontend component maps. A new runtime service contributes `service.yml` and `describe.json`; a Studio-local operator is added to `f8studio_server.studio_runtime`; a presentation renderer is added to the Web presentation registry. This keeps dependencies visible to static analysis and packaging.

## Automation

`StudioAutomationTools` is the shared boundary for deterministic agents and provider-backed agents. The HTTP API, `studio_cli`, and `studio_mcp` call the same application service and therefore use identical revision, idempotency, approval, and event semantics. Provider credentials are read only by the server process.

## Distribution

The production Web bundle is embedded in the `f8studio-server` wheel. The startup script installs the locked Pixi runtime and starts the server on loopback. The server opens the browser after Uvicorn finishes startup and binds its sockets. No compiled launcher is required. Runtime distributions install local wheels without editable source paths.

See [Build from Source](build-from-source.md) for build and verification commands.

## Generated wire contracts

Wire definitions have one source per boundary:

| Boundary | Source | Generated consumers |
| --- | --- | --- |
| Shared service/control JSON protocol | `sdk/schemas/protocol.yml` | Python msgspec models, C++ protocol models, Studio TypeScript types |
| Studio HTTP requests/responses | Server/core msgspec models and `api_contracts.ROUTES` | OpenAPI, `schemas/studio-api.gen.json`, TypeScript models and route maps |
| Event/live messages and built-in visualization payloads | `presentation_models.py` and `events.py` | TypeScript models; publishers construct the same models |
| Audio/video binary headers and format constants | `sdk/schemas/stream-wire.json` | Explicit Python/C++ header codecs |
| Runtime key templates | `sdk/schemas/runtime-keys.json` | Python/C++ key builders used by public naming APIs |
| Runtime control endpoint names | `sdk/schemas/runtime-control.json` | Python enum and C++ endpoint constants/registration list |
| Rungraph fingerprint normalization | `sdk/schemas/rungraph-fingerprint.json` | Python/C++ normalization and canonical serialization |

Run `pixi run protocol_codegen_all` after changing shared schemas or server models. CMake also generates its protocol header in the build tree and checks the checked-in stream/key/policy files. CI checks Python protocol generation, Studio contracts, stream headers, key templates and runtime policies for drift.

Generated `FooInput` models describe accepted requests: defaulted fields may be omitted. Generated `Foo` models describe emitted responses: ordinary defaults are present, while `UNSET` and `omit_defaults` preserve optionality. Recursive references and discriminator mappings stay within their input/output direction. `ApiRequests` binds browser JSON body builders to route keys; `contracts.ts` is an alias/validation facade without independent wire interfaces.

Generation does not replace boundary validation. Browser validators and server decoders still validate untrusted input. User-defined state, custom service payloads and extension presentation commands remain JSON values because their schemas are defined at runtime. UI rendering state, leases and normalized display projections remain local types. Key validation, wildcard/path normalization, and transport behavior remain explicit application code around the generated formats. Legacy command paths keep their existing spelling.

Compile-only TypeScript tests check defaults, nullability, recursive schemas, operation tags, request bodies and fixed-size coordinates. Cross-language tests compile a C++ fixture and compare its encoded bytes with the historical Python wire layout, including negative timestamps and large 64-bit identifiers.

C++ JSON models use concrete nested structs, maps, vectors, enums and recursive schema variants. Optional fields reject malformed values; optional nullable fields distinguish absence from explicit null. Parsing commits the output only on success. Explicitly dynamic `F8JsonValue` payloads remain JSON. These decoders check shapes, types and constants, not every JSON Schema numeric/string constraint. Unsupported schema constructs and conflicting enum definitions fail generation. Python/C++ fingerprint tests share a fixture including Unicode, ordering and ignored UI metadata.

## Static checks and shared lifecycle helpers

`pixi run --locked -e build-check typecheck` performs strict checking of Platform, Studio, media components, and their integration tools. Individual extensions own their implementation type checks. Missing third-party stubs remain exempt. `pixi run lint` checks E4/E7/E9/F and Bugbear rules; `lint_imports` checks package boundaries and `quality_exceptions` prevents broad/silent exception regressions.

All seven built-in Viz nodes share typed presentation injection and configuration conversion. Wave, track and 3D views share one throttled refresh owner that coalesces pending updates, reports background failures and finishes shutdown before detach. Audio/video gateway managers share typed negotiation, source leases, disconnected-session reaping and shutdown through `SessionManager`; media-specific tracks, quality settings, overlays and drop accounting stay in their concrete managers. Shutdown cancels in-flight negotiations and attempts all session cleanup before reporting failures.

## Agent and graph workspace responsibilities

`AgentService` is the public facade for session CRUD, provider selection and run task lifecycle. `AgentSessions` owns persisted records, the shared per-session locks and event publication. `AgentToolExecution` owns approval futures, revision checks, operation auditing and terminal transitions; model-driven and deterministic workflows use the same executor. `AgentModelTools.bind` creates run-local preview/proposal collections, while `DeterministicWorkflow` contains the offline workflow. Neither depends on the service facade. Wire summaries and conversation evidence are pure functions in `agents/evidence.py`.

`GraphWorkspace` composes the view and owns selection/Inspector sizing. `useGraphProject` owns authoritative project snapshots, event synchronization and the edit queue; its queue is the single source for pending mutation counts. `useGraphCanvas` owns React Flow projections and gesture handlers, with pure move/resize patch construction in `layoutEdits.ts`. `useProjectDeployment` owns deployment subscriptions and cancellation, `useGraphCommands` owns command dialogs/in-flight commands/toasts, and `GraphInspectors` owns field/hotkey presentation. Hooks receive explicit typed inputs; no component receives an opaque workspace object or calls back into another component's private methods.

Refactoring acceptance is based on state ownership, dependency direction and behavior preservation. Main-file line counts describe navigation improvements; they do not measure total source reduction. New imports, constructor signatures and regression tests count toward the overall maintenance cost.
