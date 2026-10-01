"""Validate product --describe contracts from the just-built C++ executables."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCTS = (
    "f8cppengine_service", "f8cvkit_tracking_service", "f8cvkit_dense_optflow_service",
    "f8cvkit_flow_metric_service", "f8cvkit_template_match_service", "f8cvkit_video_stab_service",
    "f8audiocap_service", "f8implayer_service", "f8screencap_service",
)


def service_executables(index_path: Path) -> list[Path]:
    from f8pysdk.service_runtime_tools.inventory.index import indexed_entry, read_service_index

    index_path = index_path.resolve()
    index = read_service_index(index_path)
    products: dict[str, Path] = {}
    for item in index.services:
        entry = indexed_entry(index_path, index, item)
        if entry is None or entry.launch.command in {"pixi", "pixi.exe"}:
            continue
        executable = Path(entry.launch.command)
        if executable.stem in PRODUCTS:
            if executable.stem in products:
                raise ValueError(f"Duplicate native product: {executable.stem}")
            products[executable.stem] = executable
    missing = set(PRODUCTS) - products.keys()
    if missing:
        raise ValueError(f"Missing native products in {index_path}: {sorted(missing)}")
    return [products[name] for name in PRODUCTS]


def validate_executable_paths(executables: list[Path]) -> None:
    missing = [str(path) for path in executables if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "C++ contract executables are missing; build/deploy services first. "
            "For --bin-dir, use the active CMake output directory. Missing: " + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bin-dir', type=Path)
    source.add_argument('--service-index', type=Path, help='Validate deployed executables with their runtime DLLs')
    args = parser.parse_args()
    executables = (service_executables(args.service_index) if args.service_index else
                   [args.bin_dir / (name + ('.exe' if os.name == 'nt' else '')) for name in PRODUCTS])
    validate_executable_paths(executables)
    with tempfile.TemporaryDirectory(prefix='f8-cpp-contracts-') as temporary:
        root = Path(temporary)
        for executable in executables:
            executable = executable.resolve()
            print(f'Validate C++ contract: {executable}', flush=True)
            try:
                result = subprocess.run([str(executable), '--describe'], cwd=executable.parent,
                                        capture_output=True, text=True, check=True, timeout=30)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f'C++ contract execution failed: {executable}; {exc}') from exc
            payload = json.loads(result.stdout)
            service_class = payload['service']['serviceClass']
            target = root.joinpath(*service_class.split('.')) / 'describe.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding='utf-8')
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
                        'tests/test_cppengine_operator_coverage.py', 'tests/test_cvkit_tracking_describe.py',
                        'tests/test_service_telemetry_state_contract.py'],
                       env={**os.environ, 'F8_CPP_DESCRIBE_ROOT': str(root)}, cwd=REPO_ROOT, check=True)


if __name__ == '__main__':
    main()
