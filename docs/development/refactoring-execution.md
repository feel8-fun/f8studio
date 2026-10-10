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
- [ ] Run local required checks and record results.
- [ ] Commit/push child repositories, then integration submodule pointers.
- [ ] Verify successful CI at every pushed revision; repair and rerun failures.

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
  suite passed.
- All extension SDK workflow references are synchronized to the exact SDK commit.
  Native lock metadata was refreshed without changing resolved package versions.
  Platform, WebStudio and MediaGateway now have independent Linux/Windows quality
  workflows that test sources and build wheels without publishing releases.

### Final integration verification

Pushed child revisions and workflow results will be recorded after all runs finish.
Main workspace quality and Windows integration are verified after updating its gitlinks.
