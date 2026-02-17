from __future__ import annotations

import sys
from pathlib import Path


def ensure_package_importable() -> None:
    packages_root = Path(__file__).resolve().parents[3]
    pystudio_root = packages_root / "f8pystudio"
    pysdk_root = packages_root / "f8pysdk"

    for root in (pystudio_root, pysdk_root):
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
