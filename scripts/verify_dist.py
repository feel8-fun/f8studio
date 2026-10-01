"""Verify an extracted release outside the checkout, using its own Pixi lock."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib


def verify_distribution(root: Path, *, skip_gpu_install: bool = False) -> None:
    manifest_path = root / "pixi.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    environments = manifest["environments"]
    subprocess.run(["pixi", "lock", "--check", "--manifest-path", str(manifest_path)], cwd=root, check=True)
    for name, environment in environments.items():
        distributions: list[str] = []
        for feature_name in environment["features"]:
            feature = manifest["feature"][feature_name]
            for distribution, dependency in feature.get("pypi-dependencies", {}).items():
                if isinstance(dependency, dict) and "path" in dependency:
                    wheel = dependency["path"]
                    if dependency.get("editable") or not wheel.startswith("wheels/") or not (root / wheel).is_file():
                        raise ValueError(f"Invalid release dependency: {distribution}: {dependency}")
                    distributions.append(distribution)
        if skip_gpu_install and "onnx" in environment["features"]:
            print(f"{name}: wheel paths and lock checked; GPU installation/inference not tested", flush=True)
            continue
        command = ["pixi", "run", "--locked", "--manifest-path", str(manifest_path), "-e", name]
        # Run performs the same locked installation as the launcher, in a fresh prefix.
        code = (
            "from importlib.metadata import distribution\n"
            "from pathlib import Path\n"
            "import sys\n"
            f"for name in {distributions!r}:\n"
            "    package = distribution(name)\n"
            "    assert Path(package.locate_file('')).resolve().is_relative_to(Path(sys.prefix).resolve()), name\n"
            "    direct = package.read_text('direct_url.json') or ''\n"
            "    assert '\"editable\": true' not in direct, name\n"
        )
        if name == "studio-runtime":
            code += (
                "from f8studio_server.app import default_web_dist\n"
                "bundle = default_web_dist()\n"
                "assert bundle.is_relative_to(Path(sys.prefix).resolve()), bundle\n"
                "assert (bundle / 'index.html').is_file(), bundle\n"
            )
        subprocess.run([*command, "python", "-P", "-c", code], cwd=root, check=True)
    if skip_gpu_install:
        # The interactive launcher installs every shipped environment, including CUDA.
        subprocess.run(["pixi", "run", "--locked", "--manifest-path", str(manifest_path),
                        "-e", "studio-runtime", "studio_launch", "--help"], cwd=root, check=True)
        print("Full launcher installation skipped because it includes GPU dependencies", flush=True)
        return
    # Exercise the shipped entrypoint, including its installer and argument forwarding.
    entrypoint = ["cmd.exe", "/d", "/c", "f8studio.cmd"] if os.name == "nt" else [str(root / "f8studio")]
    subprocess.run([*entrypoint, "--help"], cwd=root, check=True)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--skip-gpu-install", action="store_true",
                        help="Check GPU wheel paths/lock without installation; skip full launcher installer")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="f8-release-install-") as temporary:
        output = Path(temporary)
        shutil.unpack_archive(args.archive.resolve(), output)
        roots = list(output.glob("*/pixi.toml"))
        if len(roots) != 1:
            raise ValueError("Expected exactly one release manifest in archive")
        verify_distribution(roots[0].parent, skip_gpu_install=args.skip_gpu_install)
    print("Relocated release installation passed")


if __name__ == "__main__":
    main()
