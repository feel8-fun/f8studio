from __future__ import annotations

from f8pyengine.pyengine_service import PyEngineService


def _main(argv: list[str] | None = None) -> int:
    return PyEngineService().cli(argv, program_name="F8PyEngine")


if __name__ == "__main__":
    raise SystemExit(_main())
