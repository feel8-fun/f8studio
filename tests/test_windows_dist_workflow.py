"""Keep cold-run dependency preparation ahead of expensive application builds."""
from pathlib import Path

import pytest
import yaml

from f8pysdk.service_runtime_tools.inventory.index import indexed_entry, read_service_index
from scripts.install_services import pixi_environment, description_entry
import shlex


@pytest.mark.parametrize("job_name", ["warm-windows-build-caches", "build-windows-dist"])
def test_cache_checkpoints_and_service_environment_coverage(job_name: str) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/dist-windows.yml").read_text())
    steps = workflow["jobs"][job_name]["steps"]
    positions = {step["name"]: i for i, step in enumerate(steps) if "name" in step}
    install = steps[positions["Install locked build and service environments"]]
    args = shlex.split(install["run"])
    environments = {args[i + 1] for i, arg in enumerate(args) if arg == "-e"}
    assert environments == {"build-check", "cpp"}
    index_path = Path("config/service-index.json").resolve()
    index = read_service_index(index_path)
    for item in index.services:
        entry = indexed_entry(index_path, index, item)
        if entry is not None:
            environment = pixi_environment(description_entry(entry, build_check=True))
            assert environment is None or environment in environments, item.serviceClass
    bootstrap = "Warm Conan cache" if job_name.startswith("warm-") else "Prepare Conan dependencies"
    assert positions["Restore Pixi environments"] < positions[install["name"]] < positions["Save Pixi environments"]
    assert positions["Save Pixi environments"] < positions["Check Python service descriptions before native compilation"] < positions[bootstrap]
    assert positions["Restore Conan cache"] < positions[bootstrap] < positions["Save Conan cache"]
    for name, cache_id in (("Save Pixi environments", "pixi_cache"), ("Save Conan cache", "conan_cache")):
        save = steps[positions[name]]
        assert save["uses"] == "actions/cache/save@v5"
        assert f"steps.{cache_id}.outputs.cache-hit != 'true'" in save["if"]
        assert "github.ref_type" not in save["if"]
    if job_name == "build-windows-dist":
        assert positions["Save Conan cache"] < positions["Build runtime distribution"]


def test_quality_jobs_use_one_environment() -> None:
    workflow = yaml.safe_load(Path('.github/workflows/quality.yml').read_text())
    for name in ('python', 'web', 'release-wheels'):
        for step in workflow['jobs'][name]['steps']:
            if step.get('uses', '').startswith('prefix-dev/setup-pixi@'):
                assert step['with']['environments'] == 'build-check'
            if step.get('run', '').startswith('pixi run '):
                assert '-e build-check ' in step['run']


def test_dist_caches_only_the_two_prepared_build_environments() -> None:
    workflow = yaml.safe_load(Path('.github/workflows/dist-windows.yml').read_text())
    for job in workflow['jobs'].values():
        for step in job['steps']:
            if step.get('name') in {'Restore Pixi environments', 'Save Pixi environments'}:
                assert set(step['with']['path'].splitlines()) == {
                    '.pixi/envs/build-check', '.pixi/envs/cpp',
                }
            if step.get('run', '').startswith('pixi run '):
                args = shlex.split(step['run'])
                assert args[args.index('-e') + 1] in {'build-check', 'cpp'}


def test_dist_reuses_descriptions_and_uploads_only_offline_archive() -> None:
    workflow = yaml.safe_load(Path('.github/workflows/dist-windows.yml').read_text())
    steps = workflow['jobs']['build-windows-dist']['steps']
    build = next(step for step in steps if step.get('name') == 'Build runtime distribution')
    assert '--reuse-python-describes' in build['run']
    verify = next(step for step in steps if 'scripts/verify_dist.py' in step.get('run', ''))
    assert '--skip-gpu-install' not in verify['run']
    upload = next(step for step in steps if step.get('uses', '').startswith('actions/upload-artifact@'))
    assert upload['with']['path'] == 'build/dist/f8studio-windows-x86_64.zip'
    assert upload['with']['compression-level'] == 0
