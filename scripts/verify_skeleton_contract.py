from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from f8pysdk.motion import decode_skeleton_packet

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITYMODS_ROOT = REPO_ROOT / "extensions" / "f8unitymods"
UNITYMODS_MANIFEST = UNITYMODS_ROOT / "pixi.toml"
FIXTURE_PATH = REPO_ROOT / "sdk" / "python" / "tests" / "fixtures" / "unity_skeleton_v2.bin"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_fixture(data: bytes) -> dict[str, object]:
    packet = decode_skeleton_packet(data)
    trailer = packet.trailer
    if trailer is None or trailer.extension_version != 2:
        raise ValueError("Golden skeleton packet must contain an LMEX v2 trailer")
    identity = trailer.identity
    if identity is None or not identity.stable_key:
        raise ValueError("Golden skeleton packet must contain a stable v2 identity")
    if len(packet.bones) != trailer.total_bone_count:
        raise ValueError("Golden skeleton packet bone count does not match its trailer")
    return {
        "modelName": packet.model_name,
        "stableKey": packet.stable_key,
        "schema": packet.schema,
        "boneCount": len(packet.bones),
        "extensionVersion": trailer.extension_version,
        "exporterVersion": identity.exporter_version,
    }


def _verify_csharp_generator(expected: bytes) -> bool:
    if os.name != "nt":
        return False
    if not UNITYMODS_MANIFEST.is_file():
        raise FileNotFoundError(f"f8unitymods submodule is not initialized: {UNITYMODS_MANIFEST}")
    with tempfile.TemporaryDirectory(prefix="f8-skeleton-contract-") as temp_dir:
        generated_path = Path(temp_dir) / "unity_skeleton_v2.bin"
        subprocess.run(
            [
                "pixi",
                "run",
                "--manifest-path",
                str(UNITYMODS_MANIFEST),
                "protocol-fixture",
                str(generated_path),
            ],
            cwd=UNITYMODS_ROOT,
            check=True,
        )
        generated = generated_path.read_bytes()
    if generated != expected:
        raise ValueError(
            "C# SkeletonPacketEncoder output differs from the checked-in golden fixture: "
            f"expected={_sha256(expected)} generated={_sha256(generated)}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Unity exporter/Python skeleton binary contract.")
    parser.add_argument("--skip-csharp", action="store_true", help="Only decode the checked-in fixture.")
    args = parser.parse_args()

    fixture = FIXTURE_PATH.read_bytes()
    details = _validate_fixture(fixture)
    csharp_verified = False if args.skip_csharp else _verify_csharp_generator(fixture)
    print(
        json.dumps(
            {
                "status": "ok",
                "fixture": str(FIXTURE_PATH),
                "sha256": _sha256(fixture),
                "csharpGeneratorVerified": csharp_verified,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
