# Development workspace and distribution migration

The distribution repository assembles immutable publisher artifacts. Application extensions own
builds, locks and releases. WebStudio's frontend and backend are one mandatory
release unit; independent frontend/backend selection is not supported.

## Implementation order

1. Extract headless installation, runtime and artifact management into `f8platform`.
   Studio consumes that package; the platform must never import Studio or Gateway.
2. Add SDK application capability contracts: identity/version, contained locked environments,
   launch commands, provided/required protocols, endpoints and readiness probes.
3. Implement independent application installation, preparation, launch, stop, version
   selection and rollback. Changes cannot replace running applications or invalidate
   installed consumers. Platform shutdown stops owned processes.
4. Publish Media Gateway separately, using the SDK-owned media protocol and its own runtime.
   Publish WebStudio with its backend, domain code, protocol client and built frontend.
   Each publisher builds only its own implementation and declares library inputs.
5. Assemble application extensions alongside service/tool extensions. Preserve hashes and
   versions in the release lock. The bootstrap runtime contains the platform only.
6. Move publisher sources/workflows into independent repositories and retain local
   source checkouts for integration development. Remote publication is a separate
   step from making the implementation and artifacts reviewable locally.

## Compatibility and storage

Release versions, protocol versions and in-process ABI requirements are distinct.
Application extensions exchange versioned protocol data across process boundaries. Dependency
versions are selected by each application lock, not by a globally solved environment.
Shared-memory and native interfaces require separate explicit contracts when used.

Managed installations and shared caches live outside immutable application artifacts.
User data lives outside versioned payloads. Prior payloads remain available for
rollback. Updates select a prepared version only after compatibility checks; starting
that version verifies health before declaring it ready. Update is a stop/start
operation; active media sessions are not migrated automatically.

Official releases pin tested combinations. Custom compatible combinations remain
possible. The platform manages installation and lifecycle without requiring the
WebStudio UI. Initial platform API binds loopback and requires a persisted token.

## Acceptance checks

- Platform imports no Studio/Gateway modules and can run without their wheels.
- Same protocol names in distinct applications do not merge interpreter prefixes.
- Missing/incompatible providers fail before launch; cyclic dependencies fail clearly.
- Corrupt archives, uncontained inputs and mismatched release identities are rejected.
- A WebStudio artifact cannot omit frontend assets or mix frontend/backend versions.
- Prepare/start/stop/update/rollback work with isolated prefixes and logs.
- The publisher/assembly path does not compile other application implementations.
- Existing extension and Studio behavior passes regression tests.

## Repository roles

The current repository becomes the development workspace. `platform/`, `sdk/` and
`extensions/` contain independent source checkouts pinned for integration.
The separate sibling `f8distribution` repository consumes published artifacts only.
Applications are ordinary extensions carrying an `application` declaration, not
a separate package format. The platform bootstrap is infrastructure, not an extension.
