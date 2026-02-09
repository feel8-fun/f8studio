from __future__ import annotations


def _main(argv: list[str] | None = None) -> int:
    # Local import: keep `python -m f8pyengine.main --describe` as lightweight as possible.
    from f8pyengine.pyengine_service import PyEngineService

    return PyEngineService().cli(argv, program_name="F8PyEngine")


if __name__ == "__main__":
    raise SystemExit(_main())
