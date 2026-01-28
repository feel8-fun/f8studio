#include <gst/gst.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void print_env(const char* name) {
  const char* v = std::getenv(name);
  std::cout << name << "=" << (v ? v : "(unset)") << "\n";
}

static bool has_element(const char* name) {
  GstElementFactory* f = gst_element_factory_find(name);
  if (!f) return false;
  gst_object_unref(f);
  return true;
}

int main(int argc, char** argv) {
  gst_init(&argc, &argv);

  std::cout << "gst=" << gst_version_string() << "\n";
  print_env("GSTREAMER_ROOT");
  print_env("GST_PLUGIN_PATH");
  print_env("GST_PLUGIN_SYSTEM_PATH");
  print_env("GST_PLUGIN_SCANNER");

  const std::vector<std::string> must_have = {
      "webrtcbin",
      "nicesrc",
      "nicesink",
      "rtph264depay",
      "rtpvp8depay",
      "h264parse",
      "openh264dec",
      "videoconvert",
      "appsink",
      "appsrc",
  };

  bool ok = true;
  for (const auto& name : must_have) {
    const bool found = has_element(name.c_str());
    std::cout << "element " << name << ": " << (found ? "OK" : "MISSING") << "\n";
    ok = ok && found;
  }

  // VP8 decode: accept either the canonical VP8 decoder (typically from gst-plugins-good/libvpx)
  // or ffmpeg-backed decode (gst-libav) if present.
  const bool vp8dec = has_element("vp8dec");
  const bool avdec_vp8 = has_element("avdec_vp8");
  std::cout << "element vp8dec: " << (vp8dec ? "OK" : "MISSING") << "\n";
  std::cout << "element avdec_vp8: " << (avdec_vp8 ? "OK" : "MISSING") << "\n";
  const bool vp8_decode_ok = vp8dec || avdec_vp8;
  std::cout << "vp8 decode: " << (vp8_decode_ok ? "OK" : "MISSING") << "\n";
  ok = ok && vp8_decode_ok;

  return ok ? 0 : 2;
}
