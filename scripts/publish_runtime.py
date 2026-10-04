"""Publish one locked runtime from prebuilt wheels; compilation belongs to their repos."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
from typing import Literal

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk/python'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packages/f8studio_server'))

import msgspec

from f8pysdk.release_spec import PublishedArtifact, RuntimeCatalog, RuntimeDefinition
from f8studio_server.environment_definitions import materialize_locked_environment

from assemble_release import validate_runtime_artifact


def publish_runtime(source: Path, output: Path, *, runtime_id: str, provider_id: str,
                    version: str, abi: str, platform: Literal['linux-x86_64', 'windows-x86_64'], inputs_root: Path) -> Path:
    """Snapshot portable inputs and verify the existing lock without running a solver."""
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', runtime_id):
        raise ValueError('Invalid runtime ID')
    with tempfile.TemporaryDirectory(prefix='f8-runtime-publish-') as temporary:
        root = Path(temporary)
        workspace = root / 'runtimes' / runtime_id
        workspace.mkdir(parents=True)
        materialize_locked_environment(source.resolve(), runtime_id, workspace, inputs_root.resolve())
        config = root / 'config'
        config.mkdir()
        definition = RuntimeDefinition(runtime_id=runtime_id, provider_id=provider_id, version=version, abi=abi,
                                       manifest='${F8_PACKAGE_ROOT}/runtimes/' + runtime_id + '/pixi.toml')
        (config / 'runtime-environments.json').write_bytes(msgspec.json.encode(RuntimeCatalog(
            schema_version='f8runtimeCatalog/1', runtimes=(definition,))))
        # Validation rejects copied sources/editable dependencies. Publishers must
        # first build wheels and lock their release workspace against those wheels.
        validate_runtime_artifact(root)
        subprocess.run(['pixi', 'lock', '--manifest-path', str(workspace / 'pixi.toml'), '--check'], check=True)
        (config / 'artifact.json').write_bytes(msgspec.json.encode(PublishedArtifact(
            schema_version='f8artifact/1', artifact_id=runtime_id, version=version, kind='runtime', platform=platform,
        )))
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(root.rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(root).as_posix())
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    return output


class PublishRuntimeArguments(argparse.Namespace):
    source: Path
    output: Path
    runtime_id: str
    provider_id: str
    version: str
    abi: str
    platform: Literal['linux-x86_64', 'windows-x86_64']
    inputs_root: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-id', required=True)
    parser.add_argument('--provider-id', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--abi', required=True)
    parser.add_argument('--platform', choices=['linux-x86_64', 'windows-x86_64'], required=True)
    parser.add_argument('--inputs-root', type=Path, required=True)
    args = PublishRuntimeArguments()
    parser.parse_args(namespace=args)
    print(publish_runtime(args.source, args.output, runtime_id=args.runtime_id, provider_id=args.provider_id,
                          version=args.version, abi=args.abi, platform=args.platform, inputs_root=args.inputs_root))


if __name__ == '__main__':
    main()
