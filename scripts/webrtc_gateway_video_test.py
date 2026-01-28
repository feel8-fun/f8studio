import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional


def _require(module_name: str, pip_name: str | None = None):
    try:
        return __import__(module_name)
    except Exception:
        pkg = pip_name or module_name
        print(f"Missing Python dependency: {module_name}", file=sys.stderr)
        print(f"Install: python -m pip install {pkg}", file=sys.stderr)
        raise


def _now_ms() -> int:
    return int(time.time() * 1000)


VIDEO_SHM_MAGIC = 0xF8A11A01
VIDEO_SHM_VERSION = 1
VIDEO_FORMAT_BGRA32 = 1
_VIDEO_HEADER_STRUCT = struct.Struct("<7I4xQq2I")


@dataclass(frozen=True)
class VideoShmHeader:
    magic: int
    version: int
    slot_count: int
    width: int
    height: int
    pitch: int
    fmt: int
    frame_id: int
    ts_ms: int
    active_slot: int
    payload_capacity: int

    @property
    def header_bytes(self) -> int:
        return _VIDEO_HEADER_STRUCT.size

    @property
    def frame_bytes(self) -> int:
        return int(self.pitch) * int(self.height)

    @property
    def slot_offset_bytes(self) -> int:
        return self.header_bytes + int(self.active_slot) * int(self.payload_capacity)


def _read_video_header(buf: memoryview) -> Optional[VideoShmHeader]:
    if len(buf) < _VIDEO_HEADER_STRUCT.size:
        return None
    try:
        fields = _VIDEO_HEADER_STRUCT.unpack_from(buf, 0)
    except Exception:
        return None
    return VideoShmHeader(
        magic=fields[0],
        version=fields[1],
        slot_count=fields[2],
        width=fields[3],
        height=fields[4],
        pitch=fields[5],
        fmt=fields[6],
        frame_id=fields[7],
        ts_ms=fields[8],
        active_slot=fields[9],
        payload_capacity=fields[10],
    )


async def _wait_for_shm_frames(service_id: str, shm_bytes: int, timeout_s: float, min_frames: int) -> int:
    from multiprocessing.shared_memory import SharedMemory

    shm_name = f"shm.{service_id}.video"
    started = time.time()
    shm = None
    while time.time() - started < timeout_s:
        try:
            shm = SharedMemory(name=shm_name, create=False)
            break
        except FileNotFoundError:
            await asyncio.sleep(0.1)
    if shm is None:
        print(f"[shm] not found name={shm_name}", file=sys.stderr)
        return 2

    try:
        if shm.size < shm_bytes:
            print(f"[shm] warning: shm.size={shm.size} expected~={shm_bytes}")
        buf = shm.buf
        last_id = 0
        seen = 0
        while time.time() - started < timeout_s:
            hdr = _read_video_header(buf)
            if hdr and hdr.magic == VIDEO_SHM_MAGIC and hdr.version == VIDEO_SHM_VERSION and hdr.fmt == VIDEO_FORMAT_BGRA32:
                if hdr.frame_id != last_id and hdr.width > 0 and hdr.height > 0 and hdr.pitch > 0:
                    last_id = hdr.frame_id
                    seen += 1
                    print(
                        f"[shm] frameId={hdr.frame_id} {hdr.width}x{hdr.height} pitch={hdr.pitch} slot={hdr.active_slot}/{hdr.slot_count} ts={hdr.ts_ms}"
                    )
                    if seen >= min_frames:
                        return 0
            await asyncio.sleep(0.02)
        print(f"[shm] timeout waiting frames; seen={seen}", file=sys.stderr)
        return 3
    finally:
        try:
            shm.close()
        except Exception:
            pass


class _SyntheticVideoTrack:
    def __init__(self, width: int, height: int, fps: int):
        _require("aiortc")
        _require("av")
        _require("numpy")
        from aiortc import VideoStreamTrack  # type: ignore

        class _Track(VideoStreamTrack):
            def __init__(self, w: int, h: int, fps_: int):
                super().__init__()
                self.w = int(w)
                self.h = int(h)
                self.fps = max(1, int(fps_))
                self._i = 0

            async def recv(self):
                import av  # type: ignore
                import numpy  # type: ignore

                pts, time_base = await self.next_timestamp()
                self._i += 1

                img = numpy.zeros((self.h, self.w, 3), dtype=numpy.uint8)
                x = (self._i * 7) % self.w
                img[:, :x, 1] = 200
                img[:, x:, 2] = 200
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts = pts
                frame.time_base = time_base
                if (self._i % (self.fps * 2)) == 0:
                    print(f"[sender] producedFrames={self._i}")
                return frame

        self.track = _Track(width, height, fps)


async def _run_sender(ws_url: str, prefer_codec: str, duration_s: float, width: int, height: int, fps: int) -> int:
    websockets = _require("websockets")
    aiortc = _require("aiortc")
    from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
    from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer  # type: ignore
    from aiortc.rtcrtpsender import RTCRtpSender  # type: ignore
    from aiortc.sdp import candidate_from_sdp, candidate_to_sdp  # type: ignore

    session_id = str(uuid.uuid4())
    print(f"[sender] ws={ws_url}")
    print(f"[sender] sessionId={session_id}")

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))  # localhost-only
    done = asyncio.Event()

    video_track = _SyntheticVideoTrack(width, height, fps).track
    sender = pc.addTrack(video_track)
    transceiver = next((t for t in pc.getTransceivers() if t.sender == sender), None)

    prefer_codec = (prefer_codec or "").strip().upper()
    if transceiver and prefer_codec in ("VP8", "H264"):
        caps = RTCRtpSender.getCapabilities("video")
        wanted = [c for c in caps.codecs if c.mimeType.upper() == f"VIDEO/{prefer_codec}"]
        if wanted:
            transceiver.setCodecPreferences(wanted)
            print(f"[sender] codecPreference={prefer_codec}")
        else:
            print(f"[sender] codecPreference requested but not available: {prefer_codec}", file=sys.stderr)

    @pc.on("connectionstatechange")
    async def _on_conn_state():
        print(f"[sender] connectionState={pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            done.set()

    async with websockets.connect(ws_url, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "py-video-test", "ts": _now_ms()}))

        async def ws_rx():
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("sessionId") != session_id:
                    continue
                mtype = msg.get("type")

                if mtype == "webrtc.answer":
                    desc = msg.get("description") or {}
                    print("[sender] rx answer")
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=desc.get("sdp", ""), type=desc.get("type", "answer"))
                    )
                    continue

                if mtype == "webrtc.ice":
                    cand = msg.get("candidate")
                    if not isinstance(cand, dict):
                        continue
                    try:
                        sdp = cand.get("candidate", "")
                        if isinstance(sdp, str) and sdp.startswith("candidate:"):
                            sdp = sdp[len("candidate:") :]
                        ice = candidate_from_sdp(sdp)
                        ice.sdpMid = cand.get("sdpMid", None)
                        ice.sdpMLineIndex = int(cand.get("sdpMLineIndex", 0))
                        await pc.addIceCandidate(ice)
                    except Exception as e:
                        print(f"[sender] addIceCandidate failed: {e}", file=sys.stderr)
                    continue

                if mtype == "webrtc.stop":
                    print("[sender] rx stop")
                    done.set()
                    return

        @pc.on("icecandidate")
        async def _on_icecandidate(candidate):
            if candidate is None:
                return
            cand = {
                "candidate": "candidate:" + candidate_to_sdp(candidate),
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex,
            }
            await ws.send(json.dumps({"type": "webrtc.ice", "sessionId": session_id, "candidate": cand, "ts": _now_ms()}))

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await ws.send(
            json.dumps(
                {
                    "type": "webrtc.offer",
                    "sessionId": session_id,
                    "description": {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp},
                    "ts": _now_ms(),
                }
            )
        )

        rx_task = asyncio.create_task(ws_rx())
        stats_task = asyncio.create_task(_stats_loop(pc, done))
        try:
            await asyncio.wait_for(done.wait(), timeout=duration_s)
        except asyncio.TimeoutError:
            pass
        finally:
            rx_task.cancel()
            stats_task.cancel()
            try:
                await pc.close()
            except Exception:
                pass

    return 0


async def _stats_loop(pc, done_evt: asyncio.Event) -> None:
    while not done_evt.is_set():
        try:
            report = await pc.getStats()
            for stat in report.values():
                if getattr(stat, "type", None) != "outbound-rtp":
                    continue
                if getattr(stat, "kind", None) != "video":
                    continue
                bytes_sent = getattr(stat, "bytesSent", None)
                packets_sent = getattr(stat, "packetsSent", None)
                frames_encoded = getattr(stat, "framesEncoded", None)
                print(
                    f"[sender] outbound-rtp bytesSent={bytes_sent} packetsSent={packets_sent} framesEncoded={frames_encoded}"
                )
                break
        except Exception:
            pass
        await asyncio.sleep(1.0)


async def _start_gateway(
    exe: str, service_id: str, nats_url: str, ws_port: int, video_use_gstreamer: bool
) -> subprocess.Popen:
    args = [exe, "--service-id", service_id]
    if nats_url:
        args += ["--nats-url", nats_url]
    if ws_port:
        args += ["--ws-port", str(int(ws_port))]
    if video_use_gstreamer:
        args += ["--video-use-gstreamer"]

    env = os.environ.copy()
    env.setdefault("F8_RTC_LOG", "debug")
    env.setdefault("F8_LOG_FLUSH", "1")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    ready = asyncio.Event()

    def _drain():
        try:
            for line in proc.stdout:  # type: ignore[assignment]
                if not line:
                    continue
                sys.stdout.write("[gateway] " + line)
                sys.stdout.flush()
                if "webrtc_gateway started" in line:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.call_soon_threadsafe(ready.set)
                    except Exception:
                        pass
        except Exception:
            return

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    try:
        await asyncio.wait_for(ready.wait(), timeout=8.0)
    except asyncio.TimeoutError:
        pass
    return proc


async def run(args) -> int:
    gateway_proc = None
    if args.start_gateway:
        gateway_proc = await _start_gateway(
            args.gateway_exe, args.service_id, args.nats_url, args.ws_port, args.video_use_gstreamer
        )

    sender_task = asyncio.create_task(
        _run_sender(args.ws, args.codec, args.duration, args.width, args.height, args.fps)
    )
    shm_task = asyncio.create_task(
        _wait_for_shm_frames(args.service_id, args.shm_bytes, args.timeout, args.min_frames)
    )

    rc_sender = await sender_task
    rc_shm = await shm_task

    if gateway_proc is not None:
        try:
            gateway_proc.terminate()
        except Exception:
            pass
        try:
            gateway_proc.wait(timeout=3)
        except Exception:
            try:
                gateway_proc.kill()
            except Exception:
                pass

    if rc_sender != 0:
        return rc_sender
    return rc_shm


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end: aiortc sender -> f8webrtc_gateway -> VideoSHM")
    ap.add_argument("--ws", default="ws://127.0.0.1:8765/ws", help="Gateway websocket URL")
    ap.add_argument("--service-id", default="webrtc_gateway", help="Gateway --service-id (used for SHM name)")
    ap.add_argument("--codec", default="VP8", choices=["VP8", "H264", "vp8", "h264"], help="Preferred video codec")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=8.0, help="How long sender runs (s)")
    ap.add_argument("--timeout", type=float, default=12.0, help="Timeout waiting for SHM frames (s)")
    ap.add_argument("--min-frames", type=int, default=3, help="Frames required to pass")
    ap.add_argument("--shm-bytes", type=int, default=256 * 1024 * 1024, help="Expected SHM size (bytes)")
    ap.add_argument("--start-gateway", action="store_true", help="Start gateway process automatically")
    ap.add_argument("--gateway-exe", default=str((__import__("pathlib").Path("build/bin/f8webrtc_gateway_service.exe"))))
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--video-use-gstreamer", action="store_true", help="Start gateway with --video-use-gstreamer")
    args = ap.parse_args()

    if args.start_gateway and not os.path.exists(args.gateway_exe):
        ap.error(f"--gateway-exe not found: {args.gateway_exe}")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
