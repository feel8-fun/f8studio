from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.name == "nt", reason="POSIX entrypoint; Windows is verified in dist CI")
def test_shell_launcher_supports_spaces_arguments_and_install_failure(tmp_path: Path) -> None:
    release = tmp_path / "release with spaces"
    release.mkdir()
    launcher = release / "f8studio"
    shutil.copy2("scripts/launchers/f8studio", launcher)
    launcher.chmod(0o755)
    installer = release / "install_env.sh"
    installer.write_text('#!/bin/sh\nprintf "installed\\n" > installed\n')
    installer.chmod(0o755)
    pixi = release / "pixi"
    pixi.write_text('#!/bin/sh\ntest -f installed || exit 7\nprintf "%s\\n" "$@" > arguments\n')
    pixi.chmod(0o755)
    env = os.environ | {"PATH": str(release) + os.pathsep + os.environ["PATH"]}
    completed = subprocess.run([str(launcher), "--web-dist", "assets with spaces"], cwd=tmp_path, env=env)
    assert completed.returncode == 0
    assert (release / "arguments").read_text().splitlines() == [
        "run", "--locked", "-e", "studio-runtime", "studio_launch", "--web-dist", "assets with spaces",
    ]
    (release / "arguments").unlink()
    installer.write_text("#!/bin/sh\nexit 23\n")
    completed = subprocess.run([str(launcher)], cwd=tmp_path, env=env)
    assert completed.returncode == 23
    assert not (release / "arguments").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX bootstrap")
@pytest.mark.parametrize("downloader", ["curl", "wget"])
@pytest.mark.parametrize("failure", [None, "download", "install", "missing_binary"])
def test_shell_bootstraps_pixi_from_official_source(
    tmp_path: Path, downloader: str, failure: str | None,
) -> None:
    release = tmp_path / "release with spaces"
    release.mkdir()
    launcher = release / "f8studio"
    shutil.copy2("scripts/launchers/f8studio", launcher)
    launcher.chmod(0o755)
    installer = release / "install_env.sh"
    installer.write_text('#!/bin/sh\nprintf "installed\\n" > installed\n')
    installer.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    # No system Pixi or network tools can be discovered in this test environment.
    for name in ("dirname", "mktemp", "rm", "sh", "mkdir", "chmod", "cat"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(executable)
    official_script = tmp_path / "official-installer.sh"
    official_script.write_text(
        '#!/bin/sh\nset -eu\n'
        'test "$PIXI_NO_PATH_UPDATE" = 1\n'
        'mkdir -p "$PIXI_HOME/bin"\n'
        'cat > "$PIXI_HOME/bin/pixi" <<\'PIXI\'\n'
        '#!/bin/sh\ntest -f installed || exit 7\nprintf "%s\\n" "$@" > arguments\n'
        'PIXI\nchmod +x "$PIXI_HOME/bin/pixi"\n'
        if failure is None else '#!/bin/sh\nexit ' + ('19' if failure == 'install' else '0') + '\n'
    )
    download = tools / downloader
    download.write_text(
        '#!/bin/sh\nset -eu\n'
        'test "$4" = https://pixi.sh/install.sh\n'
        'printf "download\\n" >> downloads\n'
        + ('exit 22\n' if failure == 'download' else 'cat "$F8_TEST_INSTALLER" > "$3"\n')
    )
    download.chmod(0o755)
    env = os.environ | {
        "PATH": str(tools), "PIXI_HOME": str(tmp_path / "pixi home"),
        "F8_TEST_INSTALLER": str(official_script), "TMPDIR": str(tmp_path),
    }
    env.pop("PIXI_BIN_DIR", None)
    completed = subprocess.run(
        [str(launcher), "--no-browser"], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert completed.returncode == (0 if failure is None else 2), completed.stderr
    assert (release / "downloads").read_text() == "download\n"
    assert not list(tmp_path.glob("tmp.*")), "downloaded installer must be cleaned up"
    if failure is not None:
        assert not (release / "installed").exists()
        assert not (release / "arguments").exists()
        assert "Pixi" in completed.stderr
    else:
        assert (release / "arguments").read_text().splitlines() == [
            "run", "--locked", "-e", "studio-runtime", "studio_launch", "--no-browser",
        ]
        # A second launch must reuse Pixi from its install directory, without downloading again.
        subprocess.run([str(launcher)], cwd=tmp_path, env=env, check=True)
        assert (release / "downloads").read_text() == "download\n"
