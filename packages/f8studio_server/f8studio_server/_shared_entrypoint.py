"""Start a shared-runtime extension in its own process, with isolated CLI settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('python_root', type=Path)
    parser.add_argument('module')
    parser.add_argument('service_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    sys.argv = [args.module, *args.service_args]
    runpy.run_module(args.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
