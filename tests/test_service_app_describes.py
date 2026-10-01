from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module_name", "expected_service_class"),
    [
        ("f8pydl.main_classifier", "f8.dl.classifier"),
        ("f8pydl.main_detector", "f8.dl.detector"),
        ("f8pydl.main_detsorter", "f8.dl.detsorter"),
        ("f8pydl.main_humandetector", "f8.dl.humandetector"),
        ("f8pydl.main_optflow", "f8.dl.optflow"),
        ("f8pydl.main_tcnwave", "f8.dl.tcnwave"),
        ("f8pyaudiofeat.main_core", "f8.audiofeat.core"),
        ("f8pyaudiofeat.main_rhythm", "f8.audiofeat.rhythm"),
        ("f8pymppose.main_pose", "f8.mp.pose"),
        ("f8pyscript.main_expr", "f8.pyexpr"),
        ("f8pyscript.main_script", "f8.pyscript"),
        ("f8pyengine.main", "f8.pyengine"),
        ("f8proclauncher.main", "f8.proclauncher"),
    ],
)
def test_service_entrypoint_cli_describe_smoke(
    module_name: str,
    expected_service_class: str,
) -> None:
    # Integration tests belong to the superbuild, not to the standalone SDK.
    # Describe runs in a fresh process and must not depend on test import order.
    package = module_name.split('.', 1)[0]
    env = os.environ.copy()
    source_path = str(REPO_ROOT / 'extensions' / package)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (source_path, env.get('PYTHONPATH', ''))))
    result = subprocess.run(
        [sys.executable, '-m', module_name, '--describe'],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["service"]["serviceClass"] == expected_service_class
