"""Validate product --describe contracts from the just-built C++ executables."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PRODUCTS = (
    "f8cppengine_service", "f8cvkit_tracking_service", "f8cvkit_dense_optflow_service",
    "f8cvkit_flow_metric_service", "f8cvkit_template_match_service", "f8cvkit_video_stab_service",
    "f8audiocap_service", "f8implayer_service", "f8screencap_service",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='f8-cpp-contracts-') as temporary:
        root = Path(temporary)
        for name in PRODUCTS:
            executable = args.bin_dir / (name + ('.exe' if os.name == 'nt' else ''))
            result = subprocess.run([str(executable.resolve()), '--describe'],
                                    capture_output=True, text=True, check=True, timeout=30)
            payload = json.loads(result.stdout)
            service_class = payload['service']['serviceClass']
            target = root.joinpath(*service_class.split('.')) / 'describe.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding='utf-8')
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
                        'tests/test_cppengine_operator_coverage.py', 'tests/test_cvkit_tracking_describe.py',
                        'tests/test_service_telemetry_state_contract.py'],
                       env={**os.environ, 'F8_CPP_DESCRIBE_ROOT': str(root)}, check=True)


if __name__ == '__main__':
    main()
