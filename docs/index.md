# F8Studio Documentation

F8Studio documentation is organized around practical usage: setup, module reference, and complete scenarios.

- **Version**: `latest`
- **Last updated**: `2026-02-25`
- **Audience**: engineers integrating Feel8 services and runtime graphs.

## Start Here

1. Go to [Installation and Deployment](getting-started/install-deploy.md).
2. Choose your workflow:
   - [Studio (GUI)](getting-started/studio.md)
   - [Runner (Headless)](getting-started/runner.md)
3. Review module details in [Modules Overview](modules/index.md).
4. Follow end-to-end examples in [Scenarios](scenarios/index.md).

## Documentation Principles

- Markdown-first source files.
- Generated service reference from `service.yml` + `describe.json`.
- Manual usage guidance kept alongside generated pages.
- Build checks (`zensical build`) and independent nav/link validation on pull requests.

## Deployment Target

This site is designed for **Cloudflare Pages** (Git-connected, automatic publish on default branch).
