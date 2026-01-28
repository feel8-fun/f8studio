# GStreamer (Conan) WebRTC Path

This repo supports a second receive path based on `webrtcbin` (GStreamer) to reduce custom RTP/decoder surface area.

## Quick probe

Run:

`powershell -ExecutionPolicy Bypass -File scripts/run_gst_webrtc_probe.ps1`

Expected output:
- `element webrtcbin: OK`
- `element nicesrc: OK`
- `element nicesink: OK`
- `element rtph264depay: OK`
- `element openh264dec: OK`

If you see `MISSING`, the corresponding GStreamer plugin isn't being found (check `GST_PLUGIN_PATH` printed by `f8gst_probe`).

## Gateway integration

The gateway supports a runtime switch:

- `build/bin/f8webrtc_gateway_service.exe --video-use-gstreamer`

Notes:
- `webrtcbin` requires `nicesrc/nicesink` from `libnice` built with `with_gstreamer=True`. The Conan install in this repo enables that automatically and exports `gstnice.dll` into `GST_PLUGIN_PATH` via the `gst-plugins-bad` recipe.
- Run the gateway inside the Conan runenv so plugin discovery works (or set `GST_PLUGIN_PATH`/`GST_PLUGIN_SCANNER` manually):
  - `. build/generators/conanrun.ps1`

## End-to-end test

Run:

`powershell -ExecutionPolicy Bypass -File scripts/run_webrtc_gateway_video_test.ps1 -UseGstreamer`
