from __future__ import annotations

import json
from importlib import metadata
import platform

import aiortc
import av
import fastapi
import uvicorn
import zenoh


def distribution_version(name: str) -> str:
    return metadata.version(name)


def main() -> None:
    codecs = av.codecs_available
    video_encoders = [name for name in ("h264", "libx264", "vp8", "libvpx", "vp9", "libvpx-vp9") if name in codecs]
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "aiortc": aiortc.__version__,
            "av": av.__version__,
            "eclipse-zenoh": distribution_version("eclipse-zenoh"),
            "fastapi": fastapi.__version__,
            "uvicorn": uvicorn.__version__,
        },
        "imports": {
            "RTCPeerConnection": aiortc.RTCPeerConnection.__name__,
            "zenohSession": zenoh.Session.__name__,
        },
        "availableVideoEncoders": video_encoders,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
