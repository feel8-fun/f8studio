"""Description generation must not import inference implementations."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('service', ['classifier', 'detector', 'humandetector', 'optflow', 'detsorter', 'tcnwave'])
def test_describe_does_not_import_inference_modules(service: str) -> None:
    code = '''
import importlib.abc
import runpy
import sys
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'onnxruntime', 'cv2', 'f8pydl.service_node',
                        'f8pydl.optflow_service_node', 'f8pydl.tcnwave_service_node',
                        'f8pydl.detection_sorter_service_node'}:
            raise AssertionError('Description imported inference code: ' + fullname)
sys.meta_path.insert(0, NoInference())
sys.path.insert(0, 'extensions/f8pydl')
module = sys.argv[1]
sys.argv = [module, '--describe']
runpy.run_module(module, run_name='__main__')
'''
    subprocess.run([sys.executable, '-c', code, 'f8pydl.main_' + service],
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)


def test_workspace_describe_environment_excludes_gpu_dependencies() -> None:
    import yaml
    lock = yaml.safe_load(Path('pixi.lock').read_text())
    light = str(lock['environments']['build-check']['packages']).lower()
    assert 'cudnn' not in light and 'cuda-' not in light and 'onnxruntime' not in light
    extension = yaml.safe_load(Path('extensions/f8pydl/pixi.lock').read_text())
    full = str(extension['environments']['dl']['packages']).lower()
    assert 'cudnn' in full and 'onnxruntime_gpu' in full
