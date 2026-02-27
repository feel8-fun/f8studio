# Installation and Deployment

This guide sets up the project, runtime dependencies, and documentation tooling.

## Prerequisites

- Git
- Python 3.10+ (3.13 recommended)
- `pixi` for runtime environments
- Optional: C++ toolchain for native services

## Clone and Bootstrap

```bash
git clone <your-repo-url>
cd f8studio
```

Use pixi tasks for runtime commands:

```bash
pixi run -e default f8pystudio
pixi run -e default runner --help
```

Build a Studio launcher with repo icon `assets/icon.ico`:

```bash
pixi run -e ci build_studio_launcher
```
- Output is a Nuitka-built executable under `build/dist/` (platform-specific extension/name).

## Service Runtime Basics

Service launch entries are defined in `services/**/service.yml`.

Static discovery metadata is stored in `services/**/describe.json` and can be regenerated:

```bash
pixi run -e default update_describes
```

## C++ Build (Pixi + Conan + CMake)

Bootstrap dependencies and configure/build C++ services:

```bash
pixi run -e cpp cpp_bootstrap
pixi run -e cpp cpp_configure_release
pixi run -e cpp cpp_build_release
```

Refresh lockfiles when dependency versions change:

```bash
pixi lock
pixi run -e cpp cpp_lock_refresh
```

Build a distributable runtime bundle (`pixi.toml + pixi.lock + services/** + wheels/**`):

```bash
pixi run -e ci dist_ci
```

- Output directory is `build/dist/f8studio-<platform-tag>`.
- Optional archive output can be enabled with `pixi run -e ci dist_ci --archive`.

Notes:

- CI on Windows uses GitHub-hosted runner images plus `msvc-dev-cmd`; no manual Visual Studio setup is needed in CI.
- Local Windows development still requires Visual Studio Build Tools for native compilation.

## Documentation Toolchain

Install docs dependencies:

```bash
python -m pip install -r docs/requirements.txt
```

Generate and validate docs:

```bash
python scripts/check_docs_nav.py
python scripts/check_docs_links.py
zensical build
zensical serve
```

## Cloudflare Pages Build Settings

Recommended settings in Cloudflare Pages:

- **Framework preset**: None
- **Build command**:
  `python -m pip install -r docs/requirements.txt && python scripts/check_docs_nav.py && python scripts/check_docs_links.py && zensical build`
- **Build output directory**: `site`
- **Root directory**: repository root

## Note on Generated Service Pages

Service reference pages under `docs/modules/services/*.md` are generated offline from `service.yml` + `describe.json`.
Do not run generation in GitHub Pages build, because CI does not have runtime artifacts needed to produce `describe.json`.
