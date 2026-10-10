# Platform management

Feel8 Platform owns installation records, extension environments, tool jobs,
application versions and service processes. WebStudio is an optional application
and an authenticated platform client. Its frontend and backend are released together.
Closing WebStudio disconnects the editor; it does not shut down platform tools,
services or the independently managed Media Gateway.

## Development workspace

```bash
pixi run --locked platform_dev
```

Root platform tasks automatically use the minimal `workspace` environment to
prepare checkout inputs, then delegate to Launcher's own locked runtime.
`platform_tray` uses `platform-desktop`; headless tasks use `platform-runtime`.

This starts the independent platform and its small management portal. Use another
terminal to open the portal, or manage applications through the CLI:

```bash
pixi run --locked platform_open
pixi run --locked platform_cli extensions list
pixi run --locked platform_cli source start webstudio
pixi run --locked platform_cli source stop webstudio
pixi run --locked platform_stop
```

`source start webstudio` starts its declared dependencies before starting Studio.
Source commands live in each application's `config/development-application.json`.
The workspace creates the launch definitions from those files; the platform does
not special-case application identities, frontend frameworks or game engines.
Source debugging installs each application's locked Pixi environment in its own
checkout and launches it there. It never inherits Launcher's Python interpreter
and does not install an application release.
Selected application releases can provide dependencies for source debugging.
The launcher checks their declared protocol versions and resolves configured
endpoints at launch time. Stop source consumers before stopping or changing their
providers. Source and released instances of the same application cannot run together.

For desktop use, start `platform_tray` instead of `platform_dev`. The tray opens
platform management and logs, and owns the daemon's lifetime. It remains available
when WebStudio stops. If desktop support is unavailable, it runs in console mode.
Installed applications with HTTP(S) endpoints also get browser shortcuts in the
tray's **Endpoints** submenu. They are enabled only while the application is
running, and update in the background every two seconds. Source application
shortcuts are marked `(source)`.
**Restart Platform** keeps the tray open while it stops the daemon and launches
a replacement with the same arguments, port and log file. The replacement reads
the saved application startup settings. Restart and browser shortcuts are disabled
while the restart runs in the background; Exit remains available. A failed launch
is reported with a traceback and leaves Restart available for retry.
Tray polling allows up to 30 seconds for the daemon's HTTP listener to start;
initial connection refusals are recorded as startup waiting rather than warnings.
Startup timeouts and connection failures after startup retain error tracebacks.
Stop a platform started by `platform_ensure` with `platform_stop` before starting
its foreground or tray counterpart.

`studio_server` and `studio_launch` remain direct source debugging entrypoints.
They first ensure a separate platform is running, obtain the Media Gateway from
the platform and register their source instance. An external source process is
shown as externally managed; stop it in its original terminal. Platform management
will not pretend that this process is an installed artifact.

Development platform state defaults to `~/.feel8/workspaces/<workspace-path-hash>`.
`build/workspace/platform-connection.json` contains discovery information, not a
copy of the secret token. Use `--data-dir` on `platform_workspace.py` or
`F8_PLATFORM_DATA_ROOT` to choose another state root. Old Studio-owned installation
records are not automatically treated as platform-installed application releases.

## Headless use

The Portal's navigation regression can be run from the development workspace:
`pixi run --locked -e build-check node tests/browser/platform_portal_navigation.cjs`.
It requires the workspace frontend test dependencies and a Playwright Chromium
installation, or a browser selected with `F8_CHROMIUM_EXECUTABLE`. The test delays
responses across navigation and refreshes, including responses that ignore cancellation.

Extension origin and installation state are separate. All development checkouts,
including native services and tool packages, display `Development source checkout`
and their source location. A local registration does not imply a published release;
workspace presets are identified as development presets. Verified imported or
bundled release packages display their declared version and archive SHA-256.
Packages without a verified archive are identified as local packages. An application
can have both a development checkout and an imported release.

The independently installed Launcher provides the same operations without a
browser, tray, or WebStudio installation:

```bash
pixi run --locked --manifest-path platform/pixi.toml -e platform-runtime serve
```

In another terminal, run `python -m f8platform` inside that managed environment,
followed by an operation:

| Command | Purpose |
| --- | --- |
| `extensions list`, `detail ID`, `plan ID` | Inspect extension capabilities and runtime plans. |
| `extensions import URL --sha256 HASH` | Add a published extension or application package. |
| `extensions install ID`, `enable ID`, `disable ID`, `uninstall ID` | Manage installation and enablement. |
| `environments list`, `detail ID`, `prepare ID`, `cancel ID` | Inspect and prepare environments. |
| `environments storage`, `clean-unused`, `remove ID` | Inspect storage and release unused environment files while preserving the package cache. |
| `jobs list`, `show ID`, `logs ID`, `cancel ID` | Inspect or cancel queued installation and environment maintenance tasks. |
| `tools list`, `run EXTENSION TOOL --arguments args.json --confirm` | Execute declared tools. |
| `tools jobs`, `job JOB`, `cancel JOB` | Inspect or stop jobs, including continuous streams. |
| `services list`, `start INSTANCE CLASS`, `stop INSTANCE`, `logs` | Manage service processes. |
| `list`, `import ARCHIVE --sha256 HASH`, `prepare ID --sha256 HASH`, `select ID --sha256 HASH` | Manage application releases. |
| `start ID`, `stop ID`, `update ID --sha256 HASH` | Run and update selected application releases. |
| `startup --application ID` | Save the applications to start when the daemon starts. |
| `api METHOD /api/PATH --body request.json` | Access other public operations, including resources. |

Application stop requests return an accepted operation before waiting for process
exit. This allows a running application to request its own shutdown without
blocking its HTTP response. Tool and service process operations retain their own
lifecycle semantics.

Extension cards use icon buttons with hover labels for Start, Stop, Restart, Open,
installation, details and logs. Restart is available for running applications
managed by Platform, including source checkouts. It queues one operation that
waits for the old process to stop, starts the same source or selected release,
and waits for its health check. Start/Stop/Restart controls are disabled during
the operation. A stop failure prevents the new start and is recorded in task logs.
Externally managed processes must be restarted at their original entrypoint.
The restart endpoints are `POST /api/applications/{extension_id}/restart` and
`POST /api/source-applications/{extension_id}/restart`; both return a
`ManagementJob` with HTTP 202.

Package import, installation, uninstall, enablement, application release changes,
application startup and environment prepare/release requests return HTTP 202 with a
`ManagementJob`, rather than holding an HTTP request open until completion. They
use a single FIFO maintenance queue. A running installation does not prevent
submitting another task or reading inventory. Identical active requests return
the existing task ID. Preconditions are checked again when the task executes;
a failure records its error and traceback without blocking later tasks.

Use `GET /api/management-jobs` or `GET /api/management-jobs/{job_id}` to follow
`queued`, `running`, `succeeded`, `failed` and `cancelled` states. Logs are at
`GET /api/management-jobs/{job_id}/logs`. Queued tasks can be cancelled; running
tasks expose cancellation only when their installer supports it. Cancellation
returns immediately while the process shuts down in the background. Task history
survives daemon restarts. Interrupted tasks are marked failed and are not replayed
automatically, because their filesystem changes may have partially completed.

Clear completed and Dismiss remove finished maintenance records and their task
traceback files from platform storage, so they stay cleared after a daemon restart
and in other browsers. The endpoint is `POST /api/management-jobs/clear-completed`
with `{"jobIds": ["task-id", ...]}`. It rejects queued or running tasks without
clearing any requested records; already removed IDs are safe to submit again.
Clearing records does not stop applications or remove installed extensions.

The portal polls task state and updates the current page when operations complete.
Each page has a `?view=extensions|environments|tools|processes|tasks` URL, preserved
by browser reload and history navigation. Refresh reloads the current inventory,
storage snapshot and open details, and displays an update timestamp.

The API binds to loopback and requires the platform token. Portal bootstrap sets
an HTTP-only cookie; browser mutations also require the platform's own origin.
WebStudio forwards requests on the server side and does not expose the platform
credential to its JavaScript client. Public management models, route declarations,
errors and the typed HTTP client belong to the Apache-2.0 SDK.
Local browser navigation to WebStudio automatically establishes its own HTTP-only
session, including links from the platform's different loopback port. Users do not
need to find or enter a token. Cross-site requests, iframes and remote connections
cannot use this automatic bootstrap; API and WebSocket origin checks remain active.

## Ownership

The platform is the single writer of installation and environment state. Both
WebStudio and the standalone portal show that same inventory. A source checkout,
a prepared runtime, an installed release, enablement and a running process are
separate facts. Application releases use the application supervisor's records;
they are not also installed through the service/tool extension manager.

The platform rejects disabling or removing extensions whose tools or services are
running. Its process lifetime ends when the platform shuts down; disconnecting a
management client does not cancel those processes. Studio continues to own graph
editing, deployment requests, project sessions and presentation.

The independent launcher package includes the portal assets and an optional
`desktop` dependency group. The `platform-runtime` Pixi environment is headless;
`platform-desktop` adds desktop dependencies. WebStudio's release environment does
not include the launcher implementation or desktop tray dependencies.

## Environment storage

Extensions own their Pixi manifests and locks. Users do not create, derive, pin or
assign environments through the GUI or management API. The environment inspector
shows installed Conda packages and Python distributions, or locked packages when
no installed prefix exists.

Storage snapshots show logical file data and allocated file blocks. The combined
environment/cache total deduplicates hardlinks across both; adding their individual
totals would count shared data twice. Allocated block sizes are available on Unix;
Windows still reports file data and hardlink relationships. Copy-on-write extents
are not measured.

Environment lists render independently of disk usage inspection. Storage totals
are collected in one directory traversal and cached for 30 seconds, with the
measurement time included in `usageUpdatedAt`. Concurrent readers share a scan;
installation and environment maintenance invalidate the snapshot. Lifecycle
availability is recomputed even when usage totals are cached. An explicit
`GET /api/environments/storage?refresh=true` requests a new measurement, while
ordinary page navigation can reuse the snapshot. Slow inspection does not hold
the runtime lifecycle lock.

`pixi info --extended --json` reports workspace environment and cache sizes, but
also traverses directories. It does not provide a combined total that deduplicates
hardlinks across all platform-owned workspaces and their cache. Ordinary
`pixi info --json` omits sizes; `pixi list --json` can inspect locked packages,
whose download sizes are distinct from allocated filesystem blocks.

`clean-unused` releases recognized managed runtime directories that no installed
extension or retained application version references. It preserves development
checkouts, unknown directories and the shared package cache. Old developer revision
definitions are not loaded as environments or deleted automatically.

The shared Pixi package cache is retained for reuse and deduplication. Cache deletion
is not exposed through the GUI, API or CLI. Releasing environment files preserves
that cache.
