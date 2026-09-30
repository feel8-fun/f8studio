# Review remediation

The working tree contains the interrupted Claude changes; preserve them and verify each stage before continuing. Original evidence: CC_UNFINISHED.md. Review claims are not assumed to be proven, and unused browser APIs are not automatically removed.

## Stages

1. Implemented: frame epoch/default-port/dist fixes; race-free gateway startup; test baseline and CI. Full Python: 1009 passed, 7 explicitly gated C++ artifact checks skipped. Strict Studio types, lint, and three import contracts pass. Compiled artifact checks were subsequently completed; see final validation below.
2. Implemented and tested: serialized graph edits; approval timeout/denial cleanup and terminal-state protection; shared reconnecting browser event stream; recoverable media hubs and explicit shared Zenoh session ownership. Targeted server/media 54 passed, SDK ownership/transport 11 passed, frontend 90 passed.
3. Implemented and tested: reliable resumable event log with hello/cursor semantics; separate bounded latest-value mirror for monitor/state/presentation; full TCode snapshots; event-driven editor/deploy refresh; detach and pending-task cleanup. Python and web strict types pass. Unknown incremental extension commands stay reliable; REST APIs with external callers stay public.
4. Resources implemented: SQLite connections explicitly commit/rollback/close under a shared policy; bounded undo; atomic catalog replacement; per-data-dir instance locks; X11 command ownership; stable hotkey registrations; process task cleanup; per-session Agent locks; editor session locks, limits and idle reaper, with no destructive constructor cleanup. Server/core full suite 188 passed; later resource/media followup 43 passed.
5. Additional bug fixes implemented: decision superseded/expired requests emit an error for their exec ID (12 tests passed); NumPy video/audio conversions preserve pitch/nonfinite/rounding semantics (gateway 28 passed, strict types clean). Exception metrics now 110 broad / 0 silent; typed error-result propagation is explicitly recognized and caller traceback logging retained.
6. Implemented: API error/access boundary, generated contract foundation, skeleton resource reuse, stale discovery invalidation, and SDK callback/retained-state correctness. Full regression, compiled artifacts and isolated startup were validated; exact counts and remaining architectural scope are recorded below.

## Deliberate compatibility decisions

- Preserve standalone GraphStore idempotency; repository persistence has a separate lifetime and API contract.
- Existing SQLite schema is unchanged; a new schema migration is unnecessary for connection ownership alone. Legacy metadata remains compatible.
- Do not remove publicly callable REST routes merely because the browser does not call them.
- Refactor duplicate code only when input/output and failure semantics agree; file length alone is not a bug.
- Expression evaluator returns a typed result retaining the actual exception and traceback; the service boundary reports it with deduplication. This is not a swallowed exception.

## Baseline (before new changes)

- Frame source + dist: 36 passed.
- Server + media gateway: 166 passed.
- Default Python typecheck: clean.
- Root tests: 83 passed, 2 failed (ctest path expectation; C++ describe coverage based on local cache).

## Review qualifications

- Approval denial has an outer cancellation handler; permanent waiting is not established for every provider. Pending approval/tool cleanup on timeout is missing.
- DL runtime exceptions already report traceback with repeat suppression.
- Service start/stop/status/active routes have probe callers; retain their API.
- Latest-value coalescing needs explicit semantics for extension commands; do not discard arbitrary incremental operations.
- C++ flow/motion declarations include pending operators; lack of cached specs must not be confused with working implementations.
- Current source declarations omit Python script (distinct CPython counterpart), decision, and FBX skeleton player. The six flow/motion pending specs are declarations only. Adding C++ implementations is outside remediation of stale discovery/coverage tests.
- Built C++ contract tests require F8_CPP_DESCRIBE_ROOT pointing at freshly generated service descriptions; normal Python CI checks source declaration coverage without local build caches.

## Final validation and additional remediation (2026-09-29)

- Full Python suite with `F8_CPP_DESCRIBE_ROOT=/tmp/f8-review-built-describes`: **1039 passed, no skipped tests**, one third-party Scapy/cryptography deprecation warning. This run preceded the final small import-revision/expression-policy followups; those were tested separately below.
- Final Agent/API/Zenoh followup suite: **72 passed**. Earlier final API/Agent/discovery suite: **70 passed**. Expression policy/service/operator suite: **69 passed**. Media/access followup: **31 passed**.
- Web: **94 passed**, TypeScript build clean. Python basic and strict Studio checks clean; lint, all 3 import contracts, and exception budget **110 broad / 0 silent** pass.
- Rebuilt `build/Release` from current source. Product C++ describe contracts: **10 passed** using fresh binary output. CTest: **1/1 executable passed**. Added `scripts/quality/check_cpp_contracts.py` to Windows distribution CI so these checks no longer silently skip release validation.
- Isolated real HTTP smoke (temporary data directory and OS-assigned ports): authenticated browser bootstrap, unauthorized API rejection, exact Origin rejection, project create/delete, managed gateway startup, resumable events, live mirror, and clean shutdown all passed. Existing user's Studio processes were not restarted.
- Protocol Python generation successfully ran in the declared `ci` environment, to a temporary output, with a single documented datamodel-code-generator API path. Fixed obsolete generator option and removed known-attribute introspection. Studio JSON Schema/TS generation and `--check` pass.

### New confirmed bugs addressed after the resource stage

- Owner-local retained reads now agree between production Zenoh and in-memory transport. A new downstream target receives the last observed remote value from the state router, rather than relying on an impossible cluster-wide local read.
- Zenoh subscriptions/queryables use native callbacks plus an asyncio inbox instead of 1 ms polling. Queries explicitly drop after response so consolidation completes. Callback failure tests and real two-service roundtrips pass. `set_rungraph` failures now log traceback.
- Both NumPy expression languages share a numeric member allowlist. File I/O/native-library access is rejected; this is deliberately not presented as process isolation for arbitrary Python script nodes.
- Approval resolution checks layout revision as well as graph revision. Both Agent patch application paths reject editing existing code fields outside the explicit code tools. A layout-only conflict regression passes.
- Import requests from the browser carry the latest graph/layout revisions; the project lock protects the comparison and replacement. Legacy callers may omit revisions for API/1 compatibility. Stale layout import regression passes.
- Skeleton objects update transforms/buffers in place; removed mesh/line/axis/box/label resources are disposed. A WebGL-independent ownership/reuse test covers this.
- API clients share error parsing, including validation lists and no-content deletes; false runtime results throw centrally. Explicit domain input/not-found errors map to 4xx; incidental ValueError maps to a logged 500. Error responses expose code/message and preserve API/1 detail compatibility, including Retry-After.
- Production `__main__` enables random per-start credentials in private `access.json`; exact trusted origins apply to HTTP/WS. Local browser bootstrap uses an HttpOnly SameSite cookie; remote browsers have a token login form. CLI/MCP clients and probes read origin-scoped credentials, launched engine requests inherit credentials only for the matching server origin. Embedded `create_app` callers opt in using `StudioAccess`; tests do not cause a production `testserver` bypass.
- Development static describe files require a matching source fingerprint; bundles without source retain authoritative static describes. Stale cache tests pass.
- Editor sessions have per-session locking, max count, periodic idle cleanup, create/close synchronization, and no destructive directory sweep. DB connection ownership is centralized without changing existing schemas. Hotkey and ordinary state writes now share the deployment eligibility rule.

### Review suggestions deliberately not equated with bug fixes

The review mixes defects with architectural proposals. The following are **not claimed as completed rewrites**:

- OpenAPI now has an explicit request/response/status contract for every HTTP API, generated from the actual Python models. The JSON Schema/TS generator includes these route dependencies, with drift and decoder-alignment tests. Frontend response aliases cover media/runtime/editor/assets/local capabilities and status enums. Flexible graph/spec editor view models remain hand-authored projections; universal Python/C++ wire-code generation is not claimed. Existing compiled contract tests verify real product descriptions.
- `ProjectLifecycle` now owns delete/stop/restart/import/restore, and `ProjectCommits` shares mutation ordering, hotkey refresh and graph publication across lifecycle and ordinary edits. Audio/video RTC pools now share one implementation with explicit source types and retain their public APIs. Splitting Agent/GraphWorkspace purely by line count, unifying all viz factories/naming modules/C++ entrypoints remains an optional structural proposal, not a verified behavioral defect.
- Merging all expression engines, removing standalone GraphStore idempotency, removing public REST/overlay routes, deleting the old session compiler, or replacing command-state wiring. These have compatibility/semantic implications and are not justified solely by duplicate code or absent browser callers.
- The deterministic graph-builder provider remains an explicit offline/test-capable provider; changing default product/provider behavior is separate from approval lifecycle correctness.
- MCP graph editing remains an authenticated automation API, like CLI/REST. It does not pretend to be a Studio Agent approval session; the MCP host's user-approval policy governs tool invocation.
- SQLite per-component migration machinery was not introduced because no on-disk schema changed. Existing schema version handling and legacy compatibility are preserved.
- NumPy conversion removes per-pixel Python overhead but is not a claim of 60 FPS at 1080p on all hardware. A single local measurement was about 0.54 s flow / 0.10 s scalar before the final HSV array simplification; performance tuning beyond this vectorization remains measurable future work.

### Demo CLI disposition

The demo never implemented `--describe`: its abort was an uncaught cxxopts unknown-option exception, not a failed product description contract. Its process boundary now reports unsupported/invalid options and exits 2; startup exceptions report and exit 1. The silent logger-initialization catch was removed. Rebuilt executable checks: `--help` exits 0; `--describe` and an unknown option both exit 2 with actionable diagnostics. All nine product binaries retain their real description contracts.

## Completion followup

- Shared browser events isolate consumer callback failures (event/resync/connection), preserve delivery to other projections, and recover from WebSocket constructor failures without losing unsubscribe cleanup.
- Generic RTC pool preserves negotiations during the release grace period, prevents old leases from evicting replacement sessions after `closeAll`, and reclaims late answers. Audio teardown now uses keepalive like video. Existing public audio/video pool APIs are retained.
- Lifecycle operations no longer embed orchestration in HTTP handlers. Domain conflicts/unavailable errors map explicitly to 409/503. Common per-project async coordination covers patch/history/import/restore/stop/restart/delete; idle locks are weakly retained.
- Graph events are published after runtime state synchronization, so their `runtimeErrors` agree with the response. Cancelled synchronization still publishes the persisted graph with an explicit cancellation error. Idempotent retries do not duplicate events.
- Agent approvals reject a graph/layout change before showing the approval, using the exact proposed layout revision for patch/proposal/code writes. Terminal runs cannot create a fresh pending approval.
- Explicit HTTP contracts preserve API/1 casing and public routes; schema references, full route coverage, decoder/model alignment and generation drift are checked. Dynamic modding-tool dictionaries remain JSON values by design.
- Validation results for this followup are recorded after the final regression below.


### Followup validation

- Frontend full suite: **99 passed**; TypeScript clean.
- All **87** registered HTTP API routes have explicit contracts. Route coverage, schema references and actual request-decoder alignment tests pass; generated schema/TS `--check` passes.
- Python basic and strict Studio type checks, lint, all **3** import contracts, and exception budget (**110 broad / 0 silent**) pass. `git diff --check` is clean.
- Demo rebuild/CLI exit behavior and CTest (**1/1**) pass.
- Real isolated HTTP/gateway startup, authentication, exact Origin enforcement, project create/delete, event/live WebSockets and shutdown pass. User's existing services were not restarted.
- The timeout-cleanup regression now reschedules the real asyncio timeout after pending approval is observed, instead of assuming SQLite work finishes within 200 ms. Agent suite: **27 passed**.
- Final full Python regression with fresh compiled-description artifacts enabled: **1045 passed, 0 skipped**, one third-party Scapy/cryptography deprecation warning, **88.23 s**. Log: `/tmp/f8-review-completion-tests-final.log`.

## Comprehensive wire type generation (2026-09-29)

The inherited workspace was committed first as `c7d400c0` (`fix: complete review remediation and runtime lifecycle hardening`). This followup completes wire type generation; large-file decomposition is a separate task.

- Replaced handwritten browser wire interfaces with generated aliases, including graph/spec/Agent models, recursive value schemas, monitor data, media settings and skeleton scenes. Browser JSON request builders use the generated route-to-body map.
- Shared schema generation distinguishes accepted `FooInput` values from serialized `Foo` output, preserving ordinary defaults, `UNSET`, `omit_defaults`, recursive references and discriminator mappings. Schema keywords inside user defaults/field names remain data.
- All 87 HTTP routes participate in generated contracts. TypeScript/JSON Schema include 261 model definitions in each input/output direction, including every shared protocol struct and the built-in presentation models. Event/live publishers and visualization operators use explicit models; browser live parsing and visualization consumers reference generated types.
- `schemas/stream-wire.json` generates Python/C++ binary header codecs and constants, adopted by both SDK transports. Existing wire sizes, field ordering, signed timestamps and 64-bit identities are preserved.
- `schemas/runtime-keys.json` generates shared key builders, adopted by public naming APIs and runtime configuration keys. Validation/path normalization remain explicit; legacy command APIs preserve their existing paths.
- `pixi run protocol_codegen_all` regenerates shared protocol models, stream codecs, key builders and Studio contracts. CI checks all checked-in generated artifacts; CMake generates protocol models and checks stream/key drift.
- Added compile-only TS rejection tests, serializer schema tests, generated-file drift tests, and a compiled C++/Python historical-byte compatibility fixture. Dynamic extension/user payloads remain JSON by design; UI state/render projections are local types, not wire contracts.

Validation:

- Full Python suite: **1053 passed, 0 skipped** (`/tmp/f8-types-full-tests-final.log`). The final schema-keyword edge case was subsequently verified with the schema/contract/API subset: **8 passed**.
- Frontend: **99 passed**, TypeScript clean.
- SDK Python type check and strict Studio Python type check: **0 errors**.
- C++ SDK and test executable rebuilt; CTest **1/1 passed**. Compiled cross-language codec fixture and public naming compatibility tests pass.
- Python protocol regeneration matches tracked output (timestamp excluded); protocol/Studio/stream/key drift checks pass.
- Lint, all 3 import contracts, exception budget (110 broad / 0 silent), and `git diff --check` pass.
- Generation coverage and commands are documented in `docs/developers/web-studio-architecture.md`.

## Static checks, generated boundaries and shared Viz/gateway logic (2026-09-29)

- SDK/engine/service Pyright moved from basic mode with disabled diagnostics to standard mode. Argument, assignment, call, general type and optional-member errors are enabled; Studio/media retain strict mode. Ruff now enables E4/E7/E9/F and Bugbear. Fixed the newly exposed issues with explicit signatures, narrowing and public reexports. Third-party stub absence remains exempt; two local SciPy suppressions explain the installed stub mismatch. This is not a claim that the whole repository uses strict mode or contains no `Any`.
- C++ protocol generation now emits concrete nested objects, maps, vectors, enums and recursive schema variants. Malformed optional values reject, failed parses preserve the previous output, and optional nullable fields retain absent/null distinction. Unsupported typed constructs and conflicting enums fail generation rather than silently becoming JSON. User/extension payloads intentionally remain `F8JsonValue`; the parser is not a complete JSON Schema constraint validator.
- Added array `minItems`/`maxItems` to the shared schema and regenerated Python/Studio contracts, retaining the wave-expression pair-size contract.
- Control endpoint names and rungraph normalization rules now have shared schema sources with generated Python/C++ consumers. Fingerprints agree on ignored UI metadata and collection ordering; shared fixtures cover recursive types, Unicode and cross-language normalization. CMake/CI/codegen tasks include runtime policy checks.
- All seven Viz nodes share configuration readers and explicit presentation injection/factories. Wave/track/3D share owned throttled refresh scheduling, cancellation, failure reporting and shutdown-before-detach. Boolean strings and nonfinite numeric configuration are handled consistently; video uses the spec's `fit` default.
- Audio/video gateways now share a generic typed session manager for negotiation, leases, disconnect grace/reconnection and shutdown. Closing cancels in-flight offers, prevents late session registration and attempts all session cleanup before propagating errors. Track implementations, video quality/overlays and audio drop accounting remain concrete.
- Added regressions for scheduling coalescence, immediate/delayed replacement, close synchronization, failure recovery, config conversion, negotiation cancellation/failure, disconnect grace reset and partial shutdown failure. Corrected an unrelated flaky frontend fixture whose reconnect query returned a different model from session creation.
- Large-file responsibility decomposition remains separate from this task.

Validation:

- SDK standard and Studio/media strict checks: **0 errors**; lint, all **3** import contracts and exception checks (**109 broad / 0 silent**) pass.
- Viz/gateway/Studio stage regression: **203 passed**. New focused generation/lifecycle regressions: **13 passed**.
- Frontend after fixture correction: **99 passed**, TypeScript clean.
- C++ SDK/tests and engine rebuilt; CTest **1/1 passed**, containing **31** GoogleTest cases.
- Python protocol, Studio contract, stream/key/policy generated-file checks pass. `git diff --check` passes.
- First full Python run alongside C++ compilation: **1064 passed**, one language-server completion timeout. An isolated final run is recorded below when complete.
- Final isolated full Python regression with fresh product descriptions: **1065 passed, 0 skipped**, one third-party Scapy/cryptography deprecation warning, **87.35 s** (`/tmp/f8-final-full-tests-clean.log`). The completion timeout did not recur.
