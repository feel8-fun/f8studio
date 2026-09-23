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

The production Web bundle is embedded in the `f8studio-server` wheel. The launcher installs the `studio-runtime` Pixi environment, starts the server on loopback, waits for `/api/health`, and opens the browser. Runtime distributions install local wheels without editable source paths.

See [Build from Source](build-from-source.md) for build and verification commands.
