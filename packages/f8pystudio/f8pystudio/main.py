from __future__ import annotations

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        raw = (os.environ.get("F8_LOG_LEVEL") or "").strip().upper()
        if raw:
            level = getattr(logging, raw, logging.INFO)
        else:
            timings = (os.environ.get("F8_DISCOVERY_LOG_TIMINGS") or "").strip().lower()
            level = logging.INFO if timings in ("1", "true", "yes", "on", "enable", "enabled") else logging.WARNING
        logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="F8PyStudio")
    parser.add_argument("--describe", action="store_true", help="Output the service description in JSON format")
    parser.add_argument(
        "--discovery-live",
        action="store_true",
        help="Disable static describe.json/inline describe fast-paths; always run describe subprocesses.",
    )
    args = parser.parse_args(argv)

    from .pystudio_program import PyStudioProgram

    if args.discovery_live:
        os.environ["F8_DISCOVERY_DISABLE_STATIC_DESCRIBE"] = "1"

    prog = PyStudioProgram()
    if args.describe:
        print(prog.describe_json_text())
        return 0
    return prog.run()


if __name__ == "__main__":
    raise SystemExit(main())
