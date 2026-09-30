"""Keep cold-run dependency preparation ahead of expensive application builds."""
from pathlib import Path
import shlex
import tomllib

import pytest
import yaml

from f8pysdk.service_runtime_tools.inventory.index import indexed_entry, read_service_index
from scripts.install_services import pixi_environment


@pytest.mark.parametrize("job_name", ["warm-windows-build-caches", "build-windows-dist"])
def test_cache_checkpoints_and_service_environment_coverage(job_name: str) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/dist-windows.yml").read_text())
    steps = workflow["jobs"][job_name]["steps"]
    positions = {step["name"]: i for i, step in enumerate(steps) if "name" in step}
    install = steps[positions["Install locked build and service environments"]]
    args = shlex.split(install["run"])
    assert "--all" in args and "-e" not in args
    with Path("pixi.toml").open("rb") as source:
        environments = set(tomllib.load(source)["environments"])
    assert "--locked" in args
    assert {"ci", "cpp", "studio-runtime", "web-studio"} <= environments
    index_path = Path("config/service-index.json").resolve()
    index = read_service_index(index_path)
    for item in index.services:
        entry = indexed_entry(index_path, index, item)
        if entry is not None:
            environment = pixi_environment(entry)
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
