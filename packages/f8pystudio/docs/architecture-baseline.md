# F8PyStudio Architecture Baseline

## Scope
This baseline documents the current runtime-critical flows for `f8pystudio` before large-scale refactoring.

## Core Runtime Flows
1. Discovery:
- Entry: `f8pystudio.pystudio_program.PyStudioProgram.run`.
- Source: `service_catalog.discovery.load_discovery_into_registries`.
- Output: `ServiceCatalog` service/operator specs.

2. Node Class Assembly:
- Entry: `PyStudioProgram.build_node_classes`.
- Mechanism: map service/operator specs to render node classes via `RenderNodeRegistry`.
- Safety: register fallback missing operator/service node classes.

3. Graph Compile and Deploy:
- Compile entry: `nodegraph.runtime_compiler.compile_runtime_graphs_from_studio`.
- Deploy entry: `widgets.main_window.F8StudioMainWin._on_deploy_action_triggered`.
- Runtime bridge: `pystudio_service_bridge.PyStudioServiceBridge.deploy`.

4. Service Lifecycle and Process Control:
- Local process manager: `service_process_manager.ServiceProcessManager`.
- Remote lifecycle RPC: NATS endpoints `status`, `activate/deactivate`, `set_rungraph`, `set_state`, `terminate`.

5. State Roundtrip:
- UI -> Runtime: `main_window._on_ui_property_changed` calls `set_local_state` or `set_remote_state`.
- Runtime -> UI: `RemoteStateWatcher` emits `UiCommand(command="state.update")`.

6. Session Persistence:
- Graph load/save entry: `nodegraph.node_graph.F8StudioGraph.load_session` and inherited save path.
- Current target format: wrapped session envelope `f8studio-session/2`.
- One-time migration helper: `python -m f8pystudio.session_migration <session.json>`.

7. Extension Loading:
- Entry: `pystudio_program.PyStudioProgram.run`.
- Source: `F8PYSTUDIO_PLUGINS` (comma-separated module names).
- Output: plugin manifests registered into `extensions.ExtensionRegistry`, renderer registrations applied to `RenderNodeRegistry`.

## Current Coupling Hotspots
1. `nodegraph/service_basenode.py`
2. `widgets/node_property_widgets.py`
3. `pystudio_service_bridge.py`
4. `nodegraph/operator_basenode.py`
5. `nodegraph/node_graph.py`

## Compatibility Contract
1. Keep wire semantics with `f8pysdk/f8cppsdk`:
- endpoint subjects and request envelopes
- KV naming and state payload shape
2. Preserve hybrid service support:
- Python services and C++ services must both remain deployable and state-observable.
