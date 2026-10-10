# Repository quality refactoring execution

Authorized scope: implement the reviewed simplification and modularization work,
test it, push each repository, and verify GitHub CI for the pushed revisions.
Cloud Library v2 contract interoperability was rechecked and is already working.

## Work ledger

- [x] SDK: unify expression value/validation semantics and frame identity.
- [x] Pose: reset sampling/model state on stream epoch changes.
- [x] PyEngine/Studio/PyScript: consume the shared expression core.
- [x] C++ Engine: reject pending operators before execution and hide them from authoring.
- [x] Diagnostics/Unity/root: use the SDK skeleton codec; remove duplicate decoders.
- [x] DL: share telemetry, model resolution and error reporting components.
- [x] CVKit: share frame processing statistics.
- [x] SDK/Screencap/IMPlayer: share the BGRA frame sink adapter.
- [x] SDK ServiceBus: narrow owner dependencies and correct public API documentation.
- [x] WebStudio: split route registration by domain; isolate legacy migrations.
- [x] Platform: extract registration/catalog/install responsibilities and migrations.
- [x] Cloud: isolate legacy/domain routes and repositories; fix legacy console pagination/request ownership.
- [x] UnityMods: split detection, profiles, artifacts and installation.
- [x] IMPlayer: isolate media source resolution and playlist control.
- [x] ProcLauncher: share process primitives and add meaningful lifecycle tests.
- [x] Quality: unique SDK tool namespace, strict typed boundaries, actionable exception handling.
- [x] CI: independent workflows for every repository, synchronized SDK revisions, integration coverage.
- [x] Run local required checks and record results.
- [x] Commit/push child repositories, then integration submodule pointers.
- [x] Verify successful CI for the child revisions and assembled runtime; repair failures.

## Validation notes

Starting snapshot: clean root branch `dev`, clean submodules on `main`.
GitHub CLI authentication is available. Existing local Cloud sandbox is on 8787;
verification that writes data uses isolated temporary test instances.

## Completion evidence

Record commit IDs, workflow run URLs, and limitations here as work completes.

### Completed implementation and local verification

- SDK `a977f137b8280db8566bd52a62a1805092725fb9`: routing transport and
  metrics are injected, command output uses an explicit callback, and error
  deduplication has its own owner. State routing still coordinates the shared
  state pipeline; it no longer reaches into bus transport/configuration fields.
  Python 308 passed / 3 optional integrations skipped; lint, strict types and wheel passed.
  CI: https://github.com/feel8-fun/f8sdk/actions/runs/38019397586 (Linux/Windows passed).
- Cloud `dbcd5dad870ef5eaceda9cd160fc65b93494d909`: static modules separate
  authentication, HTTP, frontend and legacy asset domains. Mixed asset pagination
  and stale list/content responses have regression coverage. Full `npm run check` passed.
  CI: https://github.com/feel8-fun/f8assetcloud/actions/runs/38019282662.
- WebStudio: domain routes, graph/component/provider migrations, and shared
  expression evaluation are separate owners. Standalone strict types passed;
  core/server 404 passed; frontend 274 passed; production build passed. The
  cross-repository contract-generation and assembled workspace tests now live
  in the root workspace. Project selection has request ownership regression
  coverage; late initial loads and old selections cannot replace newer ones.
  End-to-end tests and Playwright configuration are included in strict TypeScript
  checks. Default operator spacing accounts for port hit areas, and container
  minimum width is derived from the same geometry.
- Platform: standalone strict types and 76 tests passed. Windows daemon startup
  polling accepts connection timeouts as well as refused connections.
- MediaGateway: standalone strict types and 33 tests passed.
- IMPlayer: media source parsing/authentication and playlist control are separate
  translation units. Service builds and native tests passed. Screencap and
  CVKit consume the SDK BGRA sink and monitor accumulation respectively.
- Consumers local: PyEngine/Pose + expression integration 333 passed; Pose
  restart regression 19 passed; DL 116 passed and typecheck passed; all five
  CVKit service binaries built and CVKit tests passed; ProcLauncher lifecycle
  plus Unity checks 19 passed. SDK, CVKit and IMPlayer native test suites passed.
- Workspace Linux CI: 658 passed / 8 optional checks skipped.
  Strict types, lint, import boundaries, contracts and exception checks passed.
  Workspace SDK replacement integration and installed-wheel startup passed.
  Windows integration uncovered default-codepage reads, temporary-path aliases
  and a POSIX-only permission assertion; those checks now use explicit UTF-8,
  resolved paths and the correct platform semantics. The affected 43-test local
  suite passed. Full Windows CI: 653 passed / 13 platform or optional checks
  skipped; workspace SDK replacement, strict types, frontend and installed-wheel
  startup passed.
- All extension SDK workflow references are synchronized to the exact SDK commit.
  Native lock metadata was refreshed without changing resolved package versions.
  Platform, WebStudio and MediaGateway now have independent Linux/Windows quality
  workflows that test sources and build wheels without publishing releases.

### Final integration verification

- Final WebStudio `6d03ed202130b6d4ed07b9c4c449d368632a2140` passed standalone
  Linux and Windows quality checks, including strict types, core/server tests,
  frontend tests/build and wheels.
- Full assembled browser suite: **81 passed, 11 skipped**, desktop and mobile.
  Skips cover the opt-in real-model scenario and desktop-only interaction checks.
  Real Cloud Worker login, publication, discovery and fixed-version insertion
  passed in both browser layouts. The existing debug Worker on 8787 remained
  available; integration tests used isolated Workers.
- Main implementation `9ad0f068b78420a9289f71951959a7f5acf2dce1` passed all four
  quality jobs, including full Windows integration:
  https://github.com/feel8-fun/f8studio/actions/runs/38021298825.
- Final gitlink/evidence updates use that same quality workflow. The completion
  response links the exact final workspace run. Push debounce remains configured
  for 12 hours; explicit workflow dispatch runs the checks immediately.

### Final child revisions

Every workflow below completed successfully for the listed commit.

| Repository | Commit | Successful CI |
| --- | --- | --- |
| [feel8-fun/f8sdk](https://github.com/feel8-fun/f8sdk) | [a977f137b8280db8566bd52a62a1805092725fb9](https://github.com/feel8-fun/f8sdk/commit/a977f137b8280db8566bd52a62a1805092725fb9) | [Passed](https://github.com/feel8-fun/f8sdk/actions/runs/38019397586) |
| [feel8-fun/f8assetcloud](https://github.com/feel8-fun/f8assetcloud) | [dbcd5dad870ef5eaceda9cd160fc65b93494d909](https://github.com/feel8-fun/f8assetcloud/commit/dbcd5dad870ef5eaceda9cd160fc65b93494d909) | [Passed](https://github.com/feel8-fun/f8assetcloud/actions/runs/38019282662) |
| [feel8-fun/f8platform](https://github.com/feel8-fun/f8platform) | [d77d939e849e3381e8053d211d87199cf31048e0](https://github.com/feel8-fun/f8platform/commit/d77d939e849e3381e8053d211d87199cf31048e0) | [Passed](https://github.com/feel8-fun/f8platform/actions/runs/38019728630) |
| [feel8-fun/f8audiocap](https://github.com/feel8-fun/f8audiocap) | [7894610515ad9fe61c0b43939373713a4aac89ac](https://github.com/feel8-fun/f8audiocap/commit/7894610515ad9fe61c0b43939373713a4aac89ac) | [Passed](https://github.com/feel8-fun/f8audiocap/actions/runs/38019623802) |
| [feel8-fun/f8cppengine](https://github.com/feel8-fun/f8cppengine) | [82d985185471491cb7172789209a05c10e220bb4](https://github.com/feel8-fun/f8cppengine/commit/82d985185471491cb7172789209a05c10e220bb4) | [Passed](https://github.com/feel8-fun/f8cppengine/actions/runs/38019624325) |
| [feel8-fun/f8cvkit](https://github.com/feel8-fun/f8cvkit) | [3b045c0dc7e53e9da327a96bd9a13739e201d66b](https://github.com/feel8-fun/f8cvkit/commit/3b045c0dc7e53e9da327a96bd9a13739e201d66b) | [Passed](https://github.com/feel8-fun/f8cvkit/actions/runs/38019626118) |
| [feel8-fun/f8diagnostics](https://github.com/feel8-fun/f8diagnostics) | [b7de2c6a9f5388d7333c98a8f05112f756f79058](https://github.com/feel8-fun/f8diagnostics/commit/b7de2c6a9f5388d7333c98a8f05112f756f79058) | [Passed](https://github.com/feel8-fun/f8diagnostics/actions/runs/38019486976) |
| [feel8-fun/f8implayer](https://github.com/feel8-fun/f8implayer) | [305e8a9d3f446b7fc91351ddb2b2b4e37cab6dad](https://github.com/feel8-fun/f8implayer/commit/305e8a9d3f446b7fc91351ddb2b2b4e37cab6dad) | [Passed](https://github.com/feel8-fun/f8implayer/actions/runs/38019626425) |
| [feel8-fun/f8mediagateway](https://github.com/feel8-fun/f8mediagateway) | [1060f1af9edab931d16de3601be13e2fb7acb2f8](https://github.com/feel8-fun/f8mediagateway/commit/1060f1af9edab931d16de3601be13e2fb7acb2f8) | [Passed](https://github.com/feel8-fun/f8mediagateway/actions/runs/38019500426) |
| [feel8-fun/f8proclauncher](https://github.com/feel8-fun/f8proclauncher) | [29d1abda55c9b690a50d33ccd20e2bb31efec785](https://github.com/feel8-fun/f8proclauncher/commit/29d1abda55c9b690a50d33ccd20e2bb31efec785) | [Passed](https://github.com/feel8-fun/f8proclauncher/actions/runs/38019489027) |
| [feel8-fun/f8pyaudiofeat](https://github.com/feel8-fun/f8pyaudiofeat) | [aec998a2f863fdf41a636ea3f775ef2f10832db1](https://github.com/feel8-fun/f8pyaudiofeat/commit/aec998a2f863fdf41a636ea3f775ef2f10832db1) | [Passed](https://github.com/feel8-fun/f8pyaudiofeat/actions/runs/38019490000) |
| [feel8-fun/f8pydl](https://github.com/feel8-fun/f8pydl) | [41309cdd41d010cb9045ffb2ea5a8991bf622e81](https://github.com/feel8-fun/f8pydl/commit/41309cdd41d010cb9045ffb2ea5a8991bf622e81) | [Passed](https://github.com/feel8-fun/f8pydl/actions/runs/38019493901) |
| [feel8-fun/f8pyengine](https://github.com/feel8-fun/f8pyengine) | [9548fb106158903725e4734e873754b51e694d2e](https://github.com/feel8-fun/f8pyengine/commit/9548fb106158903725e4734e873754b51e694d2e) | [Passed](https://github.com/feel8-fun/f8pyengine/actions/runs/38019494596) |
| [feel8-fun/f8pymppose](https://github.com/feel8-fun/f8pymppose) | [e80ae5574f8437f811e59451111788923715d570](https://github.com/feel8-fun/f8pymppose/commit/e80ae5574f8437f811e59451111788923715d570) | [Passed](https://github.com/feel8-fun/f8pymppose/actions/runs/38019495948) |
| [feel8-fun/f8screencap](https://github.com/feel8-fun/f8screencap) | [d1f2ccf8cf72c9f12fd1e8e194cd951602a8494c](https://github.com/feel8-fun/f8screencap/commit/d1f2ccf8cf72c9f12fd1e8e194cd951602a8494c) | [Passed](https://github.com/feel8-fun/f8screencap/actions/runs/38019628004) |
| [feel8-fun/f8unity_mods](https://github.com/feel8-fun/f8unity_mods) | [bbd3b39a04f115f5b9249713e4a2d09a61c58061](https://github.com/feel8-fun/f8unity_mods/commit/bbd3b39a04f115f5b9249713e4a2d09a61c58061) | [Passed](https://github.com/feel8-fun/f8unity_mods/actions/runs/38019497910) |
| [feel8-fun/f8webstudio](https://github.com/feel8-fun/f8webstudio) | [6d03ed202130b6d4ed07b9c4c449d368632a2140](https://github.com/feel8-fun/f8webstudio/commit/6d03ed202130b6d4ed07b9c4c449d368632a2140) | [Passed](https://github.com/feel8-fun/f8webstudio/actions/runs/38022330625) |
