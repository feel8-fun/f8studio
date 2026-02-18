from __future__ import annotations

import logging
import os

def _main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        raw = (os.environ.get("F8_LOG_LEVEL") or "").strip().upper()
        level = getattr(logging, raw, logging.WARNING) if raw else logging.WARNING
        logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

    # Local import: keep `python -m f8pyengine.main --describe` as lightweight as possible.
    from f8pyengine.pyengine_service import PyEngineService

    return PyEngineService().cli(argv, program_name="F8PyEngine")


if __name__ == "__main__":
    raise SystemExit(_main())
