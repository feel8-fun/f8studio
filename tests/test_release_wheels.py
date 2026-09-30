from __future__ import annotations

from pathlib import Path
import subprocess
from unittest import mock

import pytest

from scripts.release_wheels import build_wheels


def test_build_stages_web_assets_without_modifying_editable_source(tmp_path: Path) -> None:
    source = tmp_path / "packages" / "f8studio_server"
    package = source / "f8studio_server"
    old_bundle = package / "web_dist"
    old_bundle.mkdir(parents=True)
    (old_bundle / "index.html").write_text("old source bundle")
    (package / "__init__.py").write_text("")
    (source / "pyproject.toml").write_text("[project]\nname = 'f8studio-server'\n")
    assets = tmp_path / "build" / "web-studio"
    assets.mkdir(parents=True)
    (assets / "index.html").write_text("new release bundle")
    stage = tmp_path / "build" / "stage"
    wheels = tmp_path / "build" / "wheels"

    with mock.patch("scripts.release_wheels.subprocess.run") as run:
        build_wheels([source], wheels_dir=wheels, staging_dir=stage, web_bundle=assets)
    staged = stage / "f8studio_server"
    assert (staged / "f8studio_server" / "web_dist" / "index.html").read_text() == "new release bundle"
    assert (old_bundle / "index.html").read_text() == "old source bundle"
    assert run.call_args.args[0][-1] == str(staged.resolve())

    # A build failure must propagate without altering the editable package either.
    with mock.patch("scripts.release_wheels.subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
        with pytest.raises(subprocess.CalledProcessError):
            build_wheels([source], wheels_dir=wheels, staging_dir=stage, web_bundle=assets)
    assert (old_bundle / "index.html").read_text() == "old source bundle"
