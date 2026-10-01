from __future__ import annotations

import logging

from f8pysdk.logging_utils import configure_root_logging_from_env


def _main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        configure_root_logging_from_env()

    from .proclauncher_service import build_app

    return build_app().cli(argv, program_name="F8ProcLauncher")


if __name__ == "__main__":
    raise SystemExit(_main())
