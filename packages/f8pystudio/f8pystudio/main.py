from __future__ import annotations

import argparse

from .pystudio_program import PyStudioProgram


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F8PyStudio")
    parser.add_argument("--describe", action="store_true", help="Output the service description in JSON format")
    args = parser.parse_args(argv)

    prog = PyStudioProgram()
    if args.describe:
        print(prog.describe_json_text())
        return 0
    return prog.run()


if __name__ == "__main__":
    raise SystemExit(main())
