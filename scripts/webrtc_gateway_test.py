import argparse
import asyncio
import json
import sys
import time
import uuid


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


async def run(ws_url: str, message: str, timeout_s: float) -> int:
    websockets = _require("websockets")
    aiortc = _require("aiortc")
    from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
    from aiortc.sdp import candidate_from_sdp, candidate_to_sdp  # type: ignore

    session_id = str(uuid.uuid4())
    print(f"[test] ws={ws_url}")
    print(f"[test] sessionId={session_id}")

    pc = RTCPeerConnection()
    done = asyncio.Event()
    ok = {"value": False}

    dc = pc.createDataChannel("f8-test")

    @dc.on("open")
    def _on_open():
        print("[test] datachannel open -> sending")
        try:
            dc.send(message)
        except Exception as e:
            print(f"[test] datachannel send failed: {e}", file=sys.stderr)

    @dc.on("close")
    def _on_close():
        print("[test] datachannel closed")

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "py-test", "ts": _now_ms()}))

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
                    print("[test] rx answer")
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
                        print(f"[test] addIceCandidate failed: {e}", file=sys.stderr)
                    continue

                if mtype == "webrtc.debug" and msg.get("event") == "dcMessage":
                    text = msg.get("text", "")
                    print(f"[test] rx debug echo: {text!r}")
                    ok["value"] = True
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
            await ws.send(
                json.dumps(
                    {"type": "webrtc.ice", "sessionId": session_id, "candidate": cand, "ts": _now_ms()}
                )
            )

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
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            print("[test] timeout waiting for webrtc.debug echo", file=sys.stderr)
        finally:
            rx_task.cancel()
            try:
                await pc.close()
            except Exception:
                pass

    return 0 if ok["value"] else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8765/ws", help="Gateway websocket URL")
    ap.add_argument("--message", default="ping-from-python", help="Text sent on datachannel open")
    ap.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for echo")
    args = ap.parse_args()
    return asyncio.run(run(args.ws, args.message, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
