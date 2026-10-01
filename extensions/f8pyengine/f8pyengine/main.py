from __future__ import annotations

import logging

from f8pysdk.logging_utils import configure_root_logging_from_env

def _main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        configure_root_logging_from_env()

    # Local import: keep `python -m f8pyengine.main --describe` as lightweight as possible.
    from f8pyengine.pyengine_service import build_app

    return build_app().cli(argv, program_name="F8PyEngine")


if __name__ == "__main__":
    raise SystemExit(_main())
