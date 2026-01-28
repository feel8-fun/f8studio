#include "webrtc_gateway_service.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <rtc/rtc.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <wels/codec_api.h>

#include <vpx/vpx_decoder.h>
#include <vpx/vpx_image.h>
#include <vpx/vp8dx.h>

#if defined(F8_WITH_GST_WEBRTC)
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <gst/webrtc/webrtc.h>
#endif

#include "f8cppsdk/data_bus.h"
#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"

namespace f8::webrtc_gateway {

using json = nlohmann::json;

namespace {

json schema_string() {
  return json{{"type", "string"}};
}
json schema_integer() {
  return json{{"type", "integer"}};
}
json schema_boolean() {
  return json{{"type", "boolean"}};
}
json schema_object(const json& props, const json& required = json::array()) {
  json obj;
  obj["type"] = "object";
  obj["properties"] = props;
  if (required.is_array())
    obj["required"] = required;
  obj["additionalProperties"] = false;
  return obj;
}

json state_field(std::string name, const json& value_schema, std::string access, std::string label = {},
                 std::string description = {}, bool show_on_node = false) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  if (!label.empty())
    sf["label"] = std::move(label);
  if (!description.empty())
    sf["description"] = std::move(description);
  if (show_on_node)
    sf["showOnNode"] = true;
  return sf;
}

std::string ws_url(std::uint16_t port) {
  return "ws://127.0.0.1:" + std::to_string(static_cast<unsigned>(port)) + "/ws";
}

rtc::LogLevel rtc_level() {
  const char* env = std::getenv("F8_RTC_LOG");
  if (env) {
    std::string s(env);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "verbose")
      return rtc::LogLevel::Verbose;
    if (s == "debug")
      return rtc::LogLevel::Debug;
    if (s == "info")
      return rtc::LogLevel::Info;
    if (s == "warning" || s == "warn")
      return rtc::LogLevel::Warning;
    if (s == "error")
      return rtc::LogLevel::Error;
    if (s == "fatal")
      return rtc::LogLevel::Fatal;
    if (s == "none" || s == "off")
      return rtc::LogLevel::None;
  }
  return rtc::LogLevel::Warning;
}

const char* pc_state_name(rtc::PeerConnection::State s) {
  switch (s) {
    case rtc::PeerConnection::State::New:
      return "New";
    case rtc::PeerConnection::State::Connecting:
      return "Connecting";
    case rtc::PeerConnection::State::Connected:
      return "Connected";
    case rtc::PeerConnection::State::Disconnected:
      return "Disconnected";
    case rtc::PeerConnection::State::Failed:
      return "Failed";
    case rtc::PeerConnection::State::Closed:
      return "Closed";
  }
  return "Unknown";
}

const char* ice_state_name(rtc::PeerConnection::IceState s) {
  switch (s) {
    case rtc::PeerConnection::IceState::New:
      return "New";
    case rtc::PeerConnection::IceState::Checking:
      return "Checking";
    case rtc::PeerConnection::IceState::Connected:
      return "Connected";
    case rtc::PeerConnection::IceState::Completed:
      return "Completed";
    case rtc::PeerConnection::IceState::Failed:
      return "Failed";
    case rtc::PeerConnection::IceState::Disconnected:
      return "Disconnected";
    case rtc::PeerConnection::IceState::Closed:
      return "Closed";
  }
  return "Unknown";
}

const char* gathering_state_name(rtc::PeerConnection::GatheringState s) {
  switch (s) {
    case rtc::PeerConnection::GatheringState::New:
      return "New";
    case rtc::PeerConnection::GatheringState::InProgress:
      return "InProgress";
    case rtc::PeerConnection::GatheringState::Complete:
      return "Complete";
  }
  return "Unknown";
}

const char* signaling_state_name(rtc::PeerConnection::SignalingState s) {
  switch (s) {
    case rtc::PeerConnection::SignalingState::Stable:
      return "Stable";
    case rtc::PeerConnection::SignalingState::HaveLocalOffer:
      return "HaveLocalOffer";
    case rtc::PeerConnection::SignalingState::HaveRemoteOffer:
      return "HaveRemoteOffer";
    case rtc::PeerConnection::SignalingState::HaveLocalPranswer:
      return "HaveLocalPranswer";
    case rtc::PeerConnection::SignalingState::HaveRemotePranswer:
      return "HaveRemotePranswer";
  }
  return "Unknown";
}

std::optional<std::string> json_string(const json& obj, const char* key) {
  if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string())
    return std::nullopt;
  return obj[key].get<std::string>();
}

std::optional<json> json_object(const json& obj, const char* key) {
  if (!obj.is_object() || !obj.contains(key) || !obj[key].is_object())
    return std::nullopt;
  return obj[key];
}

std::string parse_ice_ufrag(const std::string& sdp) {
  const std::string key = "a=ice-ufrag:";
  const auto pos = sdp.find(key);
  if (pos == std::string::npos) {
    return {};
  }
  auto start = pos + key.size();
  auto end = sdp.find_first_of("\r\n", start);
  if (end == std::string::npos) {
    end = sdp.size();
  }
  while (start < end && (sdp[start] == ' ' || sdp[start] == '\t')) {
    ++start;
  }
  while (end > start && (sdp[end - 1] == ' ' || sdp[end - 1] == '\t')) {
    --end;
  }
  return sdp.substr(start, end - start);
}

std::string summarize_sdp_mlines(const std::string& sdp) {
  std::string out;
  std::size_t pos = 0;
  while (pos < sdp.size()) {
    auto end = sdp.find_first_of("\r\n", pos);
    if (end == std::string::npos)
      end = sdp.size();
    std::string line = sdp.substr(pos, end - pos);
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.rfind("m=", 0) == 0 || line.rfind("a=group:BUNDLE", 0) == 0 || line.rfind("a=ice-ufrag:", 0) == 0 ||
        line.rfind("a=ice-pwd:", 0) == 0) {
      if (!out.empty())
        out += " | ";
      out += line;
    }
    pos = (end < sdp.size() && sdp[end] == '\r' && (end + 1) < sdp.size() && sdp[end + 1] == '\n') ? (end + 2)
                                                                                                     : (end + 1);
  }
  return out;
}

int mid_to_mline_index(const std::string& mid) {
  if (mid.empty())
    return 0;
  try {
    std::size_t pos = 0;
    const int v = std::stoi(mid, &pos, 10);
    if (pos == mid.size() && v >= 0)
      return v;
  } catch (...) {
  }
  return 0;
}

bool force_offer_h264_only(rtc::Description& desc, std::string& out_err) {
  out_err.clear();
  bool has_video = false;
  bool has_h264 = false;
  try {
    const int n = desc.mediaCount();
    for (int i = 0; i < n; ++i) {
      auto v = desc.media(i);
      if (!std::holds_alternative<rtc::Description::Media*>(v)) {
        continue;
      }
      auto* m = std::get<rtc::Description::Media*>(v);
      if (!m) {
        continue;
      }
      if (m->type() != "video") {
        continue;
      }
      has_video = true;

      // Keep only H264 payloads: remove other codecs and RTX/FEC.
      m->removeFormat("VP8");
      m->removeFormat("vp8");
      m->removeFormat("VP9");
      m->removeFormat("vp9");
      m->removeFormat("AV1");
      m->removeFormat("av1");
      m->removeFormat("H265");
      m->removeFormat("h265");
      m->removeFormat("RED");
      m->removeFormat("red");
      m->removeFormat("ULPFEC");
      m->removeFormat("ulpfec");
      m->removeFormat("FLEXFEC-03");
      m->removeFormat("flexfec-03");
      m->removeFormat("RTX");
      m->removeFormat("rtx");

      try {
        for (const int pt : m->payloadTypes()) {
          const auto* rm = m->rtpMap(pt);
          if (!rm) {
            continue;
          }
          std::string fmt = rm->format;
          std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                         [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
          if (fmt == "H264") {
            has_h264 = true;
            break;
          }
        }
      } catch (...) {
      }
    }
  } catch (...) {
    out_err = "parse_failed";
    return false;
  }
  if (!has_video) {
    out_err = "no_video_mline";
    return false;
  }
  if (!has_h264) {
    out_err = "no_h264_after_filter";
    return false;
  }
  return true;
}

std::optional<std::string> find_first_media_mid(rtc::Description& desc, const std::string& media_type) {
  try {
    const int n = desc.mediaCount();
    for (int i = 0; i < n; ++i) {
      auto v = desc.media(i);
      if (!std::holds_alternative<rtc::Description::Media*>(v)) {
        continue;
      }
      auto* m = std::get<rtc::Description::Media*>(v);
      if (!m) {
        continue;
      }
      if (m->type() != media_type) {
        continue;
      }
      return m->mid();
    }
  } catch (...) {
  }
  return std::nullopt;
}

std::optional<int> find_first_payload_type(rtc::Description& desc, const std::string& media_type,
                                          const std::string& codec_format_upper) {
  try {
    const int n = desc.mediaCount();
    for (int i = 0; i < n; ++i) {
      auto v = desc.media(i);
      if (!std::holds_alternative<rtc::Description::Media*>(v)) {
        continue;
      }
      auto* m = std::get<rtc::Description::Media*>(v);
      if (!m || m->type() != media_type) {
        continue;
      }
      for (const int pt : m->payloadTypes()) {
        const auto* rm = m->rtpMap(pt);
        if (!rm) {
          continue;
        }
        std::string fmt = rm->format;
        std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (fmt == codec_format_upper) {
          return pt;
        }
      }
    }
  } catch (...) {
  }
  return std::nullopt;
}

std::optional<std::string> find_fmtp_for_payload_type_from_sdp(const std::string& sdp, const std::string& media_type,
                                                               int pt) {
  if (pt < 0 || pt > 127) {
    return std::nullopt;
  }
  bool in_media = false;
  std::size_t pos = 0;
  while (pos < sdp.size()) {
    auto end = sdp.find_first_of("\r\n", pos);
    if (end == std::string::npos) {
      end = sdp.size();
    }
    std::string line = sdp.substr(pos, end - pos);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    pos = (end < sdp.size() && sdp[end] == '\r' && (end + 1) < sdp.size() && sdp[end + 1] == '\n') ? (end + 2)
                                                                                                       : (end + 1);
    if (line.rfind("m=", 0) == 0) {
      in_media = (line.rfind("m=" + media_type, 0) == 0);
      continue;
    }
    if (!in_media) {
      continue;
    }
    const std::string prefix = "a=fmtp:" + std::to_string(pt) + " ";
    if (line.rfind(prefix, 0) == 0) {
      return line.substr(prefix.size());
    }
  }
  return std::nullopt;
}

bool base64_decode(const std::string& in, std::vector<std::uint8_t>& out) {
  out.clear();
  auto b64 = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return static_cast<int>(c - 'A');
    if (c >= 'a' && c <= 'z')
      return static_cast<int>(c - 'a') + 26;
    if (c >= '0' && c <= '9')
      return static_cast<int>(c - '0') + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
    if (c == '=')
      return -2;  // padding
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
      return -3;  // whitespace
    return -1;
  };
  std::uint32_t val = 0;
  int valb = -8;
  for (unsigned char c : in) {
    const int d = b64(c);
    if (d == -3) {
      continue;
    }
    if (d == -2) {
      break;
    }
    if (d < 0) {
      return false;
    }
    val = (val << 6) + static_cast<std::uint32_t>(d);
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<std::uint8_t>((val >> valb) & 0xFFu));
      valb -= 8;
    }
  }
  return !out.empty();
}

void h264_cache_param_sets_from_annexb(const rtc::binary& annexb, rtc::binary& sps_out, rtc::binary& pps_out) {
  if (annexb.size() < 5) {
    return;
  }
  const auto* p = reinterpret_cast<const std::uint8_t*>(annexb.data());
  const std::size_t n = annexb.size();
  auto find_start = [&](std::size_t from) -> std::optional<std::pair<std::size_t, std::size_t>> {
    for (std::size_t j = from; j + 3 < n; ++j) {
      if (p[j] == 0 && p[j + 1] == 0 && p[j + 2] == 1) {
        return std::make_pair(j, j + 3);
      }
      if (p[j] == 0 && p[j + 1] == 0 && p[j + 2] == 0 && p[j + 3] == 1) {
        return std::make_pair(j, j + 4);
      }
    }
    return std::nullopt;
  };

  std::size_t i = 0;
  while (true) {
    auto s = find_start(i);
    if (!s.has_value()) {
      return;
    }
    const auto sc_pos = s->first;
    const auto nal_start = s->second;
    auto s2 = find_start(nal_start);
    const std::size_t nal_end = s2.has_value() ? s2->first : n;
    if (nal_start >= nal_end || nal_start >= n) {
      return;
    }
    const std::uint8_t nal_type = p[nal_start] & 0x1Fu;
    if (nal_type == 7) {
      sps_out.assign(annexb.begin() + static_cast<std::ptrdiff_t>(sc_pos), annexb.begin() + static_cast<std::ptrdiff_t>(nal_end));
    } else if (nal_type == 8) {
      pps_out.assign(annexb.begin() + static_cast<std::ptrdiff_t>(sc_pos), annexb.begin() + static_cast<std::ptrdiff_t>(nal_end));
    }
    i = nal_end;
    if (!s2.has_value()) {
      return;
    }
  }
}

void h264_cache_param_sets_from_fmtp(const std::string& fmtp, rtc::binary& sps_out, rtc::binary& pps_out) {
  const auto key = std::string("sprop-parameter-sets=");
  const auto p = fmtp.find(key);
  if (p == std::string::npos) {
    return;
  }
  auto v = fmtp.substr(p + key.size());
  const auto end = v.find_first_of("; \t\r\n");
  if (end != std::string::npos) {
    v = v.substr(0, end);
  }
  const auto comma = v.find(',');
  if (comma == std::string::npos) {
    return;
  }
  const auto b64_sps = v.substr(0, comma);
  const auto b64_pps = v.substr(comma + 1);

  std::vector<std::uint8_t> sps;
  std::vector<std::uint8_t> pps;
  if (!base64_decode(b64_sps, sps) || !base64_decode(b64_pps, pps)) {
    return;
  }
  auto to_annexb = [](const std::vector<std::uint8_t>& nal, rtc::binary& out) {
    out.clear();
    out.push_back(std::byte{0x00});
    out.push_back(std::byte{0x00});
    out.push_back(std::byte{0x00});
    out.push_back(std::byte{0x01});
    out.insert(out.end(), reinterpret_cast<const std::byte*>(nal.data()),
               reinterpret_cast<const std::byte*>(nal.data() + nal.size()));
  };
  to_annexb(sps, sps_out);
  to_annexb(pps, pps_out);
}

struct H264PtInfo {
  int pt = -1;
  std::string fmtp;
  int score = 0;
};

static std::optional<int> find_preferred_h264_payload_type_from_sdp(const std::string& sdp, rtc::Description& desc) {
  std::unordered_set<int> allowed_pts;
  try {
    const int n = desc.mediaCount();
    for (int i = 0; i < n; ++i) {
      auto v = desc.media(i);
      if (!std::holds_alternative<rtc::Description::Media*>(v)) {
        continue;
      }
      auto* m = std::get<rtc::Description::Media*>(v);
      if (!m || m->type() != "video") {
        continue;
      }
      for (const int pt : m->payloadTypes()) {
        allowed_pts.insert(pt);
      }
    }
  } catch (...) {
  }

  std::unordered_map<int, H264PtInfo> h264;
  bool in_video = false;

  std::size_t pos = 0;
  while (pos < sdp.size()) {
    auto end = sdp.find_first_of("\r\n", pos);
    if (end == std::string::npos) {
      end = sdp.size();
    }
    std::string line = sdp.substr(pos, end - pos);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    pos = (end < sdp.size() && sdp[end] == '\r' && (end + 1) < sdp.size() && sdp[end + 1] == '\n') ? (end + 2)
                                                                                                       : (end + 1);

    if (line.rfind("m=", 0) == 0) {
      in_video = (line.rfind("m=video", 0) == 0);
      continue;
    }
    if (!in_video) {
      continue;
    }
    if (line.rfind("a=rtpmap:", 0) == 0) {
      const auto sp = line.find(' ');
      if (sp == std::string::npos) {
        continue;
      }
      const auto colon = line.find(':');
      if (colon == std::string::npos || colon + 1 >= sp) {
        continue;
      }
      const std::string pt_str = line.substr(colon + 1, sp - (colon + 1));
      int pt = -1;
      try {
        pt = std::stoi(pt_str);
      } catch (...) {
        continue;
      }
      if (!allowed_pts.empty() && allowed_pts.find(pt) == allowed_pts.end()) {
        continue;
      }
      std::string codec = line.substr(sp + 1);
      const auto slash = codec.find('/');
      if (slash != std::string::npos) {
        codec = codec.substr(0, slash);
      }
      std::transform(codec.begin(), codec.end(), codec.begin(),
                     [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
      if (codec != "H264") {
        continue;
      }
      h264.try_emplace(pt, H264PtInfo{pt});
      continue;
    }
    if (line.rfind("a=fmtp:", 0) == 0) {
      const auto sp = line.find(' ');
      if (sp == std::string::npos) {
        continue;
      }
      const auto colon = line.find(':');
      if (colon == std::string::npos || colon + 1 >= sp) {
        continue;
      }
      const std::string pt_str = line.substr(colon + 1, sp - (colon + 1));
      int pt = -1;
      try {
        pt = std::stoi(pt_str);
      } catch (...) {
        continue;
      }
      if (h264.find(pt) == h264.end()) {
        continue;
      }
      h264[pt].fmtp = line.substr(sp + 1);
      continue;
    }
  }

  if (h264.empty()) {
    return std::nullopt;
  }

  auto score_fmtp = [](const std::string& fmtp) -> int {
    if (fmtp.empty()) {
      return 0;
    }
    const auto contains = [&](const char* s) { return fmtp.find(s) != std::string::npos; };
    int score = 0;
    if (contains("packetization-mode=1")) {
      score += 100;
    } else if (contains("packetization-mode=0")) {
      score -= 50;
    }
    const auto p = fmtp.find("profile-level-id=");
    if (p != std::string::npos) {
      auto v = fmtp.substr(p + std::string("profile-level-id=").size());
      const auto end = v.find_first_of("; \t\r\n");
      if (end != std::string::npos) {
        v = v.substr(0, end);
      }
      std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (v.rfind("42", 0) == 0) {
        score += 50;
      } else if (v.rfind("4d", 0) == 0) {
        score += 30;
      } else if (v.rfind("64", 0) == 0) {
        score += 10;
      }
    }
    if (contains("level-asymmetry-allowed=1")) {
      score += 5;
    }
    return score;
  };

  std::optional<int> best_pt;
  int best_score = std::numeric_limits<int>::min();
  for (auto& [pt, info] : h264) {
    info.score = score_fmtp(info.fmtp);
    if (!best_pt.has_value() || info.score > best_score || (info.score == best_score && pt < *best_pt)) {
      best_pt = pt;
      best_score = info.score;
    }
  }
  return best_pt;
}

struct TargetDims {
  unsigned w = 0;
  unsigned h = 0;
};

TargetDims choose_target_dims(unsigned src_w, unsigned src_h, int max_w, int max_h) {
  if (src_w == 0 || src_h == 0) {
    return {};
  }
  unsigned out_w = src_w;
  unsigned out_h = src_h;
  if (max_w > 0 || max_h > 0) {
    const double mw = max_w > 0 ? static_cast<double>(max_w) : static_cast<double>(src_w);
    const double mh = max_h > 0 ? static_cast<double>(max_h) : static_cast<double>(src_h);
    const double sx = mw / static_cast<double>(src_w);
    const double sy = mh / static_cast<double>(src_h);
    const double s = std::min(1.0, std::min(sx, sy));
    out_w = static_cast<unsigned>(std::llround(static_cast<double>(src_w) * s));
    out_h = static_cast<unsigned>(std::llround(static_cast<double>(src_h) * s));
  }
  out_w = std::max(1u, out_w);
  out_h = std::max(1u, out_h);
  return {out_w, out_h};
}

std::string hex_prefix(const rtc::binary& data, std::size_t max_bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  const std::size_t n = std::min<std::size_t>(max_bytes, data.size());
  std::string out;
  out.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t b = std::to_integer<std::uint8_t>(data[i]);
    out.push_back(kHex[(b >> 4) & 0x0F]);
    out.push_back(kHex[b & 0x0F]);
  }
  return out;
}

bool looks_like_vp8_keyframe(const rtc::binary& vp8_frame) {
  if (vp8_frame.size() < 6)
    return false;
  const std::uint8_t b0 = std::to_integer<std::uint8_t>(vp8_frame[0]);
  if ((b0 & 0x01u) != 0) {
    return false;  // interframe
  }
  // Keyframes have the start code 0x9d 0x01 0x2a at bytes 3..5.
  const std::uint8_t s0 = std::to_integer<std::uint8_t>(vp8_frame[3]);
  const std::uint8_t s1 = std::to_integer<std::uint8_t>(vp8_frame[4]);
  const std::uint8_t s2 = std::to_integer<std::uint8_t>(vp8_frame[5]);
  return (s0 == 0x9d && s1 == 0x01 && s2 == 0x2a);
}

bool looks_like_h264_idr_annexb(const rtc::binary& annexb) {
  if (annexb.size() < 5) {
    return false;
  }
  const auto* p = reinterpret_cast<const std::uint8_t*>(annexb.data());
  const std::size_t n = annexb.size();
  std::size_t i = 0;
  while (i + 4 < n) {
    std::size_t sc = std::string::npos;
    for (std::size_t j = i; j + 3 < n; ++j) {
      if (p[j] == 0 && p[j + 1] == 0 && p[j + 2] == 1) {
        sc = j + 3;
        break;
      }
      if (p[j] == 0 && p[j + 1] == 0 && p[j + 2] == 0 && p[j + 3] == 1) {
        sc = j + 4;
        break;
      }
    }
    if (sc == std::string::npos || sc >= n) {
      return false;
    }
    const std::uint8_t nal_header = p[sc];
    const std::uint8_t nal_type = nal_header & 0x1Fu;
    if (nal_type == 5) {
      return true;
    }
    i = sc + 1;
  }
  return false;
}

enum class VideoCodec { None, H264, VP8 };

class RtpPacketCounter final : public rtc::MediaHandler {
 public:
  RtpPacketCounter(std::string session_id, std::atomic<std::uint64_t>* packets, std::atomic<std::uint64_t>* bytes,
                   std::atomic<int>* last_pt)
      : session_id_(std::move(session_id)), packets_(packets), bytes_(bytes), last_pt_(last_pt) {}

  void incoming(rtc::message_vector& messages, const rtc::message_callback&) override {
    for (auto& m : messages) {
      if (!m || m->type != rtc::Message::Binary)
        continue;
      if (bytes_) {
        bytes_->fetch_add(static_cast<std::uint64_t>(m->size()), std::memory_order_relaxed);
      }

      int pt = -1;
      if (m->frameInfo) {
        pt = static_cast<int>(m->frameInfo->payloadType);
      } else if (!m->empty() && m->size() >= sizeof(rtc::RtpHeader)) {
        const auto* hdr = reinterpret_cast<const rtc::RtpHeader*>(m->data());
        if (hdr->version() == 2) {
          pt = static_cast<int>(hdr->payloadType());
        }
      }
      if (last_pt_) {
        last_pt_->store(pt, std::memory_order_relaxed);
      }

      if (packets_) {
        const auto prev = packets_->fetch_add(1, std::memory_order_relaxed);
        if (prev == 0) {
          spdlog::info("webrtc video rtp rx first sessionId={} pt={} bytes={}", session_id_, pt, m->size());
        }
      }
    }
  }

 private:
  std::string session_id_;
  std::atomic<std::uint64_t>* packets_ = nullptr;
  std::atomic<std::uint64_t>* bytes_ = nullptr;
  std::atomic<int>* last_pt_ = nullptr;
};

class Vp8RtpDepacketizer final : public rtc::MediaHandler {
 public:
  explicit Vp8RtpDepacketizer(std::vector<int> vp8_pts = {}) {
    for (const int pt : vp8_pts) {
      if (pt >= 0 && pt <= 127) {
        vp8_pts_.insert(static_cast<std::uint8_t>(pt));
      }
    }
  }

  void incoming(rtc::message_vector& messages, const rtc::message_callback&) override {
    rtc::message_vector out;
    out.reserve(messages.size());

    auto flush = [&](rtc::message_vector& dest) {
      if (!started_ || frame_.empty())
        return;
      auto fi = std::make_shared<rtc::FrameInfo>(cur_ts_);
      fi->payloadType = cur_pt_;
      dest.push_back(rtc::make_message(rtc::binary(reinterpret_cast<const std::byte*>(frame_.data()),
                                                  reinterpret_cast<const std::byte*>(frame_.data() + frame_.size())),
                                      fi));
      frame_.clear();
      started_ = false;
      have_end_ = false;
      parts_.clear();
    };

    for (auto& m : messages) {
      if (!m || m->type != rtc::Message::Binary)
        continue;
      if (rtc::IsRtcp(*m))
        continue;

      // libdatachannel may provide raw RTP packets, or already-stripped RTP payloads with FrameInfo set.
      std::uint32_t ts = 0;
      std::uint16_t seq = 0;
      std::uint8_t pt = 0;
      bool have_seq = false;
      bool marker = false;
      const std::uint8_t* payload = nullptr;
      std::size_t payload_size = 0;

      bool parsed_rtp = false;
      std::uint32_t ssrc = 0;
      if (m->size() >= sizeof(rtc::RtpHeader) && (reinterpret_cast<const rtc::RtpHeader*>(m->data())->version() == 2)) {
        const auto* hdr = reinterpret_cast<const rtc::RtpHeader*>(m->data());
        const auto hdr_size = hdr->getSize();
        if (hdr_size < m->size() && hdr_size >= sizeof(rtc::RtpHeader)) {
          ts = hdr->timestamp();
          seq = hdr->seqNumber();
          pt = hdr->payloadType();
          ssrc = hdr->ssrc();
          marker = hdr->marker() != 0;
          have_seq = true;
          payload = reinterpret_cast<const std::uint8_t*>(m->data()) + hdr_size;
          payload_size = m->size() - hdr_size;
          // Strip RTP padding if present. Padding bytes are not part of the codec bitstream.
          if (hdr->padding() != 0 && payload_size > 0) {
            const std::uint8_t pad = *(reinterpret_cast<const std::uint8_t*>(m->data()) + (m->size() - 1));
            if (pad != 0 && pad <= payload_size) {
              payload_size -= pad;
            }
          }
          // Validate against FrameInfo payloadType when present; avoid misdetecting VP8 payload as RTP.
          if (m->frameInfo && m->frameInfo->payloadType != 0 && m->frameInfo->payloadType != pt) {
            parsed_rtp = false;
          } else {
            parsed_rtp = true;
          }
        }
      }

      if (!parsed_rtp) {
        // Not a raw RTP packet. Pass through untouched (likely already depacketized upstream).
        out.push_back(std::move(m));
        continue;
      }

      if (!vp8_pts_.empty() && vp8_pts_.find(pt) == vp8_pts_.end()) {
        continue;  // ignore RTX/FEC/etc
      }

      // Lock on SSRC once we see the first RTP packet to avoid misinterpreting non-RTP payloads as RTP.
      if (!have_ssrc_) {
        ssrc_ = ssrc;
        have_ssrc_ = true;
      } else if (ssrc != ssrc_) {
        continue;
      }

      if (payload_size < 1)
        continue;

      // VP8 payload descriptor (RFC 7741).
      std::size_t idx = 0;
      const std::uint8_t b0 = payload[idx++];
      const bool x = (b0 & 0x80u) != 0;
      const bool s = (b0 & 0x10u) != 0;
      const std::uint8_t pid = (b0 & 0x0Fu);
      const bool r = (b0 & 0x40u) != 0;
      if (r) {
        continue;
      }
      if (pid > 8u) {
        continue;
      }

      if (x) {
        if (idx >= payload_size)
          continue;
        const std::uint8_t b1 = payload[idx++];
        if ((b1 & 0x0Fu) != 0) {
          continue;
        }
        const bool i = (b1 & 0x80u) != 0;
        const bool l = (b1 & 0x40u) != 0;
        const bool t = (b1 & 0x20u) != 0;
        const bool k = (b1 & 0x10u) != 0;
        if (i) {
          if (idx >= payload_size)
            continue;
          const std::uint8_t pic = payload[idx++];
          if (pic & 0x80u) {
            if (idx >= payload_size)
              continue;
            idx += 1;
          }
        }
        if (l) {
          if (idx >= payload_size)
            continue;
          idx += 1;
        }
        if (t || k) {
          if (idx >= payload_size)
            continue;
          idx += 1;
        }
      }
      if (idx >= payload_size)
        continue;

      if (!have_ts_) {
        reset_frame(ts);
      } else if (ts != cur_ts_) {
        const std::int32_t diff = static_cast<std::int32_t>(ts - cur_ts_);
        if (diff < 0) {
          // Out-of-order packet from an older timestamp/frame; ignore it rather than rewinding.
          continue;
        }
        flush(out);
        reset_frame(ts);
      }

      if (!started_) {
        // Start of partition is signaled by S=1. Most streams use PID=0 for the first partition, but be permissive.
        if (!s) {
          continue;
        }
        // For correct VP8 bitstream reconstruction, start at partition 0.
        if (pid != 0) {
          continue;
        }
        started_ = true;
        cur_pt_ = pt;
        start_seq_ = seq;
        parts_.clear();
        have_end_ = false;
        end_seq_ = seq;
      }

      const auto* vp8 = payload + idx;
      const std::size_t vp8_size = payload_size - idx;
      if (vp8_size == 0) {
        continue;
      }

      // Collect by sequence (we'll validate contiguous seq range, then reassemble by partition id).
      auto [it, inserted] = parts_.emplace(seq, Part{});
      if (inserted) {
        it->second.pid = pid;
        it->second.data.assign(vp8, vp8 + vp8_size);
      }
      if (marker) {
        have_end_ = true;
        end_seq_ = seq;
      }
      if (!have_end_)
        continue;

      // Verify we have every sequence number between start and marker (no loss). Then assemble in sequence order.
      std::uint16_t cur = start_seq_;
      for (;;) {
        auto pit = parts_.find(cur);
        if (pit == parts_.end()) {
          started_ = false;
          frame_.clear();
          have_end_ = false;
          parts_.clear();
          break;
        }
        if (cur == end_seq_) {
          frame_.clear();
          std::uint16_t cur2 = start_seq_;
          bool ok_frame = true;
          for (;;) {
            auto pit2 = parts_.find(cur2);
            if (pit2 == parts_.end()) {
              started_ = false;
              frame_.clear();
              have_end_ = false;
              parts_.clear();
              ok_frame = false;
              break;
            }
            frame_.insert(frame_.end(), pit2->second.data.begin(), pit2->second.data.end());
            if (cur2 == end_seq_) {
              break;
            }
            cur2 = static_cast<std::uint16_t>(cur2 + 1);
          }
          if (ok_frame) {
            flush(out);
          }
          break;
        }
        cur = static_cast<std::uint16_t>(cur + 1);
      }
    }

    messages.swap(out);
  }

 private:
  struct Part {
    std::uint8_t pid = 0;
    std::vector<std::uint8_t> data;
  };

  void reset_frame(std::uint32_t ts) {
    have_ts_ = true;
    cur_ts_ = ts;
    started_ = false;
    frame_.clear();
    have_end_ = false;
    parts_.clear();
    have_ssrc_ = false;
    ssrc_ = 0;
  }

  bool have_ts_ = false;
  std::uint32_t cur_ts_ = 0;
  std::uint8_t cur_pt_ = 0;
  bool started_ = false;
  bool have_end_ = false;
  std::uint16_t start_seq_ = 0;
  std::uint16_t end_seq_ = 0;
  std::unordered_map<std::uint16_t, Part> parts_;
  std::unordered_set<std::uint8_t> vp8_pts_;
  bool have_ssrc_ = false;
  std::uint32_t ssrc_ = 0;
  std::vector<std::uint8_t> frame_;
};

class H264RtpDepacketizer final : public rtc::MediaHandler {
 public:
  using FrameSink = std::function<void(rtc::binary&& data, std::uint8_t pt, std::uint32_t rtp_ts, bool is_keyframe)>;

  explicit H264RtpDepacketizer(std::vector<int> h264_pts = {}, FrameSink sink = {}) : sink_(std::move(sink)) {
    for (const int pt : h264_pts) {
      if (pt >= 0 && pt <= 127) {
        h264_pts_.insert(static_cast<std::uint8_t>(pt));
      }
    }
  }

  void incoming(rtc::message_vector& messages, const rtc::message_callback&) override {
    rtc::message_vector out;
    out.reserve(messages.size());

    auto flush = [&](rtc::message_vector& dest) {
      if (!started_ || corrupt_ || au_.empty()) {
        reset();
        return;
      }
      if (sink_) {
        const bool is_keyframe = au_has_idr_ || (au_has_sps_ && au_has_pps_);
        sink_(rtc::binary(reinterpret_cast<const std::byte*>(au_.data()),
                          reinterpret_cast<const std::byte*>(au_.data() + au_.size())),
              cur_pt_, cur_ts_, is_keyframe);
      } else {
        auto fi = std::make_shared<rtc::FrameInfo>(cur_ts_);
        fi->payloadType = cur_pt_;
        dest.push_back(rtc::make_message(
            rtc::binary(reinterpret_cast<const std::byte*>(au_.data()),
                        reinterpret_cast<const std::byte*>(au_.data() + au_.size())),
            fi));
      }
      reset();
    };

    auto start_new = [&](std::uint32_t ts, std::uint8_t pt) {
      reset();
      started_ = true;
      cur_ts_ = ts;
      cur_pt_ = pt;
    };

    auto append_start_code = [&]() {
      static constexpr std::uint8_t sc[4] = {0, 0, 0, 1};
      au_.insert(au_.end(), sc, sc + 4);
    };

    for (auto& m : messages) {
      if (!m || m->type != rtc::Message::Binary)
        continue;
      if (rtc::IsRtcp(*m))
        continue;

      if (m->size() < sizeof(rtc::RtpHeader) || (reinterpret_cast<const rtc::RtpHeader*>(m->data())->version() != 2)) {
        continue;
      }
      const auto* hdr = reinterpret_cast<const rtc::RtpHeader*>(m->data());
      const auto hdr_size = hdr->getSize();
      if (hdr_size >= m->size() || hdr_size < sizeof(rtc::RtpHeader)) {
        continue;
      }

      const std::uint32_t ts = hdr->timestamp();
      const std::uint16_t seq = hdr->seqNumber();
      const std::uint8_t pt = hdr->payloadType();
      const bool marker = hdr->marker() != 0;

      if (!h264_pts_.empty() && h264_pts_.find(pt) == h264_pts_.end()) {
        continue;
      }

      const std::uint8_t* payload = reinterpret_cast<const std::uint8_t*>(m->data()) + hdr_size;
      std::size_t payload_size = m->size() - hdr_size;
      if (hdr->padding() != 0 && payload_size > 0) {
        const std::uint8_t pad = *(reinterpret_cast<const std::uint8_t*>(m->data()) + (m->size() - 1));
        if (pad != 0 && pad <= payload_size) {
          payload_size -= pad;
        }
      }
      if (payload_size < 1) {
        continue;
      }

      if (!started_) {
        start_new(ts, pt);
      } else if (ts != cur_ts_) {
        flush(out);
        start_new(ts, pt);
      }

      if (have_seq_) {
        const std::uint16_t expected = static_cast<std::uint16_t>(last_seq_ + 1);
        if (seq != expected) {
          corrupt_ = true;
        }
      }
      last_seq_ = seq;
      have_seq_ = true;

      const std::uint8_t nal_type = payload[0] & 0x1Fu;
      if (nal_type >= 1 && nal_type <= 23) {
        if (nal_type == 5) {
          au_has_idr_ = true;
        }
        if (nal_type == 7) {
          au_has_sps_ = true;
        }
        if (nal_type == 8) {
          au_has_pps_ = true;
        }
        append_start_code();
        au_.insert(au_.end(), payload, payload + payload_size);
      } else if (nal_type == 24) {  // STAP-A
        std::size_t idx = 1;
        while (idx + 2 <= payload_size) {
          const std::uint16_t nalu_len = static_cast<std::uint16_t>((payload[idx] << 8) | payload[idx + 1]);
          idx += 2;
          if (nalu_len == 0) {
            continue;
          }
          if (idx + nalu_len > payload_size) {
            corrupt_ = true;
            break;
          }
          const std::uint8_t inner_type = payload[idx] & 0x1Fu;
          if (inner_type == 5) {
            au_has_idr_ = true;
          }
          if (inner_type == 7) {
            au_has_sps_ = true;
          }
          if (inner_type == 8) {
            au_has_pps_ = true;
          }
          append_start_code();
          au_.insert(au_.end(), payload + idx, payload + idx + nalu_len);
          idx += nalu_len;
        }
      } else if (nal_type == 28) {  // FU-A
        if (payload_size < 2) {
          corrupt_ = true;
        } else {
          const std::uint8_t fu_indicator = payload[0];
          const std::uint8_t fu_header = payload[1];
          const bool s = (fu_header & 0x80u) != 0;
          const std::uint8_t orig_type = fu_header & 0x1Fu;
          if (orig_type == 5) {
            au_has_idr_ = true;
          }
          if (orig_type == 7) {
            au_has_sps_ = true;
          }
          if (orig_type == 8) {
            au_has_pps_ = true;
          }
          const std::uint8_t nal_header = static_cast<std::uint8_t>((fu_indicator & 0xE0u) | orig_type);
          if (s) {
            append_start_code();
            au_.push_back(nal_header);
          }
          au_.insert(au_.end(), payload + 2, payload + payload_size);
        }
      }

      if (marker) {
        flush(out);
      }
    }

    messages.swap(out);
  }

 private:
  void reset() {
    au_.clear();
    started_ = false;
    corrupt_ = false;
    have_seq_ = false;
    au_has_idr_ = false;
    au_has_sps_ = false;
    au_has_pps_ = false;
  }

  std::unordered_set<std::uint8_t> h264_pts_;
  FrameSink sink_;
  std::vector<std::uint8_t> au_;
  bool started_ = false;
  bool corrupt_ = false;
  bool have_seq_ = false;
  std::uint32_t cur_ts_ = 0;
  std::uint16_t last_seq_ = 0;
  std::uint8_t cur_pt_ = 0;
  bool au_has_idr_ = false;
  bool au_has_sps_ = false;
  bool au_has_pps_ = false;
};

class LibVpxVp8Decoder {
 public:
  LibVpxVp8Decoder() = default;
  ~LibVpxVp8Decoder() { reset(); }

  LibVpxVp8Decoder(const LibVpxVp8Decoder&) = delete;
  LibVpxVp8Decoder& operator=(const LibVpxVp8Decoder&) = delete;

  void reset() {
    if (inited_) {
      try {
        (void)vpx_codec_destroy(&ctx_);
      } catch (...) {
      }
    }
    inited_ = false;
    i420_.clear();
    have_keyframe_ = false;
  }

  bool decode_bgra(const rtc::binary& vp8_frame, cv::Mat& out_bgra, int max_w, int max_h, unsigned& out_src_w,
                   unsigned& out_src_h, std::string& out_err) {
    out_err.clear();
    out_src_w = 0;
    out_src_h = 0;
    if (vp8_frame.empty())
      return (out_err = "empty_vp8"), false;

    const bool is_keyframe = looks_like_vp8_keyframe(vp8_frame);
    if (!have_keyframe_ && !is_keyframe) {
      out_err = "vp8_waiting_for_keyframe";
      return false;
    }

    if (!inited_) {
      vpx_codec_dec_cfg_t cfg{};
      cfg.threads = 0;
      cfg.w = 0;
      cfg.h = 0;
      const vpx_codec_err_t err = vpx_codec_dec_init(&ctx_, vpx_codec_vp8_dx(), &cfg, 0);
      if (err != VPX_CODEC_OK) {
        out_err = std::string("vpx_codec_dec_init failed: ") + vpx_codec_err_to_string(err);
        return false;
      }
      inited_ = true;
    }

    const auto* data = reinterpret_cast<const std::uint8_t*>(vp8_frame.data());
    const auto size = static_cast<unsigned int>(vp8_frame.size());
    const vpx_codec_err_t derr = vpx_codec_decode(&ctx_, data, size, nullptr, 0);
    if (derr != VPX_CODEC_OK) {
      out_err = std::string("vpx_codec_decode failed: ") + vpx_codec_err_to_string(derr);
      return false;
    }

    vpx_codec_iter_t it = nullptr;
    vpx_image_t* img = vpx_codec_get_frame(&ctx_, &it);
    if (!img) {
      out_err = "vpx_no_frame";
      return false;
    }
    if (img->fmt != VPX_IMG_FMT_I420 && img->fmt != VPX_IMG_FMT_I42016) {
      out_err = "vpx_unsupported_fmt";
      return false;
    }

    have_keyframe_ = true;

    const int disp_w = static_cast<int>(img->d_w);
    const int disp_h = static_cast<int>(img->d_h);
    if (disp_w <= 0 || disp_h <= 0)
      return (out_err = "vpx_bad_dims"), false;

    out_src_w = static_cast<unsigned>(disp_w);
    out_src_h = static_cast<unsigned>(disp_h);
    const TargetDims td = choose_target_dims(out_src_w, out_src_h, max_w, max_h);
    if (td.w == 0 || td.h == 0)
      return (out_err = "bad_target_dims"), false;

    // OpenCV's COLOR_YUV2* I420 conversion requires even width/height because it assumes 4:2:0 with full 2x2 chroma
    // sampling and a packed (h*3/2, w) single-channel buffer. libvpx provides both "display" (d_w/d_h) and "stored"
    // (w/h) dimensions; the display size may be odd (cropped), while the stored size is padded/aligned and safe for I420.
    int conv_w = static_cast<int>(img->w);
    int conv_h = static_cast<int>(img->h);
    if (conv_w <= 0 || conv_h <= 0) {
      conv_w = disp_w;
      conv_h = disp_h;
    }
    conv_w &= ~1;
    conv_h &= ~1;
    if (conv_w <= 0 || conv_h <= 0)
      return (out_err = "vpx_bad_conv_dims"), false;

    const int uv_w = conv_w / 2;
    const int uv_h = conv_h / 2;
    const std::size_t y_bytes = static_cast<std::size_t>(conv_w) * static_cast<std::size_t>(conv_h);
    const std::size_t uv_bytes = static_cast<std::size_t>(uv_w) * static_cast<std::size_t>(uv_h);
    const std::size_t total = y_bytes + 2 * uv_bytes;
    if (i420_.size() != total) {
      i420_.assign(total, 0);
    }

    std::uint8_t* dst_y = i420_.data();
    std::uint8_t* dst_u = dst_y + y_bytes;
    std::uint8_t* dst_v = dst_u + uv_bytes;

    const std::uint8_t* src_y = img->planes[VPX_PLANE_Y];
    const std::uint8_t* src_u = img->planes[VPX_PLANE_U];
    const std::uint8_t* src_v = img->planes[VPX_PLANE_V];
    const int stride_y = img->stride[VPX_PLANE_Y];
    const int stride_u = img->stride[VPX_PLANE_U];
    const int stride_v = img->stride[VPX_PLANE_V];
    if (!src_y || !src_u || !src_v)
      return (out_err = "vpx_missing_planes"), false;

    for (int y = 0; y < conv_h; ++y) {
      std::memcpy(dst_y + static_cast<std::size_t>(y) * conv_w, src_y + static_cast<std::size_t>(y) * stride_y, conv_w);
    }
    for (int y = 0; y < uv_h; ++y) {
      std::memcpy(dst_u + static_cast<std::size_t>(y) * uv_w, src_u + static_cast<std::size_t>(y) * stride_u, uv_w);
      std::memcpy(dst_v + static_cast<std::size_t>(y) * uv_w, src_v + static_cast<std::size_t>(y) * stride_v, uv_w);
    }

    try {
      cv::Mat yuv(conv_h + conv_h / 2, conv_w, CV_8UC1, i420_.data());
      cv::Mat bgra_full;
      cv::cvtColor(yuv, bgra_full, cv::COLOR_YUV2BGRA_I420);

      const int crop_w = std::max(1, std::min(disp_w, conv_w));
      const int crop_h = std::max(1, std::min(disp_h, conv_h));
      cv::Mat bgra = bgra_full;
      if (crop_w != conv_w || crop_h != conv_h) {
        bgra = bgra_full(cv::Rect(0, 0, crop_w, crop_h));
      }

      if (static_cast<unsigned>(bgra.cols) != td.w || static_cast<unsigned>(bgra.rows) != td.h) {
        cv::resize(bgra, out_bgra, cv::Size(static_cast<int>(td.w), static_cast<int>(td.h)), 0.0, 0.0,
                   cv::INTER_LINEAR);
      } else {
        out_bgra = bgra;
      }
      if (!out_bgra.isContinuous()) {
        out_bgra = out_bgra.clone();
      }
      return true;
    } catch (const cv::Exception& e) {
      out_err = std::string("opencv_vp8: ") + e.what();
      return false;
    } catch (...) {
      out_err = "opencv_vp8: unknown_error";
      return false;
    }
  }

 private:
  bool inited_ = false;
  vpx_codec_ctx_t ctx_{};
  std::vector<std::uint8_t> i420_;
  bool have_keyframe_ = false;
};

class OpenH264Decoder {
 public:
  OpenH264Decoder() = default;
  ~OpenH264Decoder() { reset(); }

  OpenH264Decoder(const OpenH264Decoder&) = delete;
  OpenH264Decoder& operator=(const OpenH264Decoder&) = delete;

  bool ensure() {
    if (dec_)
      return true;
    if (WelsCreateDecoder(&dec_) != 0 || !dec_) {
      dec_ = nullptr;
      return false;
    }
    SDecodingParam p{};
    p.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;
    p.bParseOnly = false;
    if (dec_->Initialize(&p) != 0) {
      reset();
      return false;
    }
    return true;
  }

  void reset() {
    if (dec_) {
      try {
        dec_->Uninitialize();
      } catch (...) {
      }
      WelsDestroyDecoder(dec_);
      dec_ = nullptr;
    }
  }

  static bool looks_like_annexb(const rtc::binary& data) {
    if (data.size() < 4)
      return false;
    const auto b0 = std::to_integer<std::uint8_t>(data[0]);
    const auto b1 = std::to_integer<std::uint8_t>(data[1]);
    const auto b2 = std::to_integer<std::uint8_t>(data[2]);
    const auto b3 = std::to_integer<std::uint8_t>(data[3]);
    return (b0 == 0x00 && b1 == 0x00 && ((b2 == 0x01) || (b2 == 0x00 && b3 == 0x01)));
  }

  static bool avcc_to_annexb(const rtc::binary& avcc, rtc::binary& out_annexb) {
    out_annexb.clear();
    if (avcc.size() < 4)
      return false;
    std::size_t off = 0;
    while (off + 4 <= avcc.size()) {
      const std::uint32_t n =
          (std::to_integer<std::uint32_t>(avcc[off]) << 24) | (std::to_integer<std::uint32_t>(avcc[off + 1]) << 16) |
          (std::to_integer<std::uint32_t>(avcc[off + 2]) << 8) | std::to_integer<std::uint32_t>(avcc[off + 3]);
      off += 4;
      if (n == 0 || off + n > avcc.size())
        return false;
      out_annexb.push_back(std::byte{0x00});
      out_annexb.push_back(std::byte{0x00});
      out_annexb.push_back(std::byte{0x00});
      out_annexb.push_back(std::byte{0x01});
      out_annexb.insert(out_annexb.end(), avcc.begin() + static_cast<std::ptrdiff_t>(off),
                        avcc.begin() + static_cast<std::ptrdiff_t>(off + n));
      off += n;
    }
    return !out_annexb.empty();
  }

  bool decode_bgra(const rtc::binary& in_data, cv::Mat& out_bgra, int max_w, int max_h, unsigned& out_src_w,
                   unsigned& out_src_h, std::string& out_err, int& out_rv) {
    out_err.clear();
    out_rv = 0;
    out_src_w = 0;
    out_src_h = 0;
    if (!ensure())
      return (out_err = "openh264_init_failed"), false;
    if (in_data.empty())
      return (out_err = "empty_frame"), false;

    rtc::binary annexb;
    if (looks_like_annexb(in_data)) {
      annexb = in_data;
    } else {
      if (!avcc_to_annexb(in_data, annexb)) {
        return (out_err = "invalid_h264_container"), false;
      }
    }

    unsigned char* planes[3] = {nullptr, nullptr, nullptr};
    SBufferInfo info{};
    std::memset(&info, 0, sizeof(info));

    const auto* in = reinterpret_cast<const unsigned char*>(annexb.data());
    const int in_size = static_cast<int>(annexb.size());

    int rv = dec_->DecodeFrameNoDelay(const_cast<unsigned char*>(in), in_size, planes, &info);
    if (rv != 0) {
      out_rv = rv;
      out_err = "openh264_decode_failed";
      return false;
    }
    if (info.iBufferStatus != 1) {
      // Some streams require a flush call to output a frame.
      std::memset(planes, 0, sizeof(planes));
      std::memset(&info, 0, sizeof(info));
      rv = dec_->DecodeFrame2(nullptr, 0, planes, &info);
      if (rv != 0 || info.iBufferStatus != 1) {
        out_rv = rv;
        out_err = "openh264_no_output";
        return false;
      }
    }

    const auto& sb = info.UsrData.sSystemBuffer;
    const int src_w = sb.iWidth;
    const int src_h = sb.iHeight;
    if (src_w <= 0 || src_h <= 0)
      return (out_err = "invalid_dimensions"), false;

    out_src_w = static_cast<unsigned>(src_w);
    out_src_h = static_cast<unsigned>(src_h);

    const TargetDims td = choose_target_dims(out_src_w, out_src_h, max_w, max_h);
    if (td.w == 0 || td.h == 0)
      return (out_err = "bad_target_dims"), false;

    // Pack I420 into a contiguous buffer (OpenCV expects planar I420 layout).
    const int stride_y = sb.iStride[0];
    const int stride_u = sb.iStride[1];
    const int stride_v = sb.iStride[1];  // openh264 uses same stride for U/V
    const int conv_w = src_w & ~1;
    const int conv_h = src_h & ~1;
    if (conv_w <= 0 || conv_h <= 0) {
      return (out_err = "invalid_dimensions_even"), false;
    }
    const int uv_w = conv_w / 2;
    const int uv_h = conv_h / 2;

    const std::size_t y_bytes = static_cast<std::size_t>(conv_w) * static_cast<std::size_t>(conv_h);
    const std::size_t uv_bytes = static_cast<std::size_t>(uv_w) * static_cast<std::size_t>(uv_h);
    i420_.resize(y_bytes + 2 * uv_bytes);
    std::uint8_t* dst_y = i420_.data();
    std::uint8_t* dst_u = dst_y + y_bytes;
    std::uint8_t* dst_v = dst_u + uv_bytes;

    const std::uint8_t* src_y = planes[0];
    const std::uint8_t* src_u = planes[1];
    const std::uint8_t* src_v = planes[2];
    if (!src_y || !src_u || !src_v)
      return (out_err = "missing_planes"), false;

    for (int y = 0; y < conv_h; ++y) {
      std::memcpy(dst_y + static_cast<std::size_t>(y) * conv_w, src_y + static_cast<std::size_t>(y) * stride_y, conv_w);
    }
    for (int y = 0; y < uv_h; ++y) {
      std::memcpy(dst_u + static_cast<std::size_t>(y) * uv_w, src_u + static_cast<std::size_t>(y) * stride_u, uv_w);
      std::memcpy(dst_v + static_cast<std::size_t>(y) * uv_w, src_v + static_cast<std::size_t>(y) * stride_v, uv_w);
    }

    try {
      cv::Mat yuv(conv_h + conv_h / 2, conv_w, CV_8UC1, i420_.data());
      cv::Mat bgra_full;
      cv::cvtColor(yuv, bgra_full, cv::COLOR_YUV2BGRA_I420);

      const int crop_w = std::max(1, std::min(src_w, conv_w));
      const int crop_h = std::max(1, std::min(src_h, conv_h));
      cv::Mat bgra = bgra_full;
      if (crop_w != conv_w || crop_h != conv_h) {
        bgra = bgra_full(cv::Rect(0, 0, crop_w, crop_h));
      }

      if (static_cast<unsigned>(bgra.cols) != td.w || static_cast<unsigned>(bgra.rows) != td.h) {
        cv::resize(bgra, out_bgra, cv::Size(static_cast<int>(td.w), static_cast<int>(td.h)), 0.0, 0.0,
                   cv::INTER_LINEAR);
      } else {
        out_bgra = bgra;
      }
      if (!out_bgra.isContinuous()) {
        out_bgra = out_bgra.clone();
      }
      return true;
    } catch (const cv::Exception& e) {
      out_err = std::string("opencv_h264: ") + e.what();
      return false;
    } catch (...) {
      out_err = "opencv_h264: unknown_error";
      return false;
    }
  }

 private:
  ISVCDecoder* dec_ = nullptr;
  std::vector<std::uint8_t> i420_;
};

}  // namespace

struct WebRtcGatewayService::WebRtcSession {
  std::string session_id;
  std::string client_id;
  std::unique_ptr<rtc::PeerConnection> pc;
  std::shared_ptr<rtc::DataChannel> dc;
  std::shared_ptr<rtc::Track> local_audio;
  std::shared_ptr<rtc::Track> local_video;
  std::shared_ptr<rtc::Track> remote_video;
  VideoCodec selected_video_codec = VideoCodec::None;
  int selected_video_pt = -1;
  bool answer_sent = false;
  std::int64_t created_ms = 0;
  std::int64_t last_msg_ms = 0;
  std::string local_ice_ufrag;
  bool use_gst = false;

#if defined(F8_WITH_GST_WEBRTC)
  GstElement* gst_pipeline = nullptr;
  GstElement* gst_webrtcbin = nullptr;
  GstElement* gst_appsink = nullptr;
  bool gst_video_branch_linked = false;
#endif

  // Video receive (compressed frames, post-depacketizer).
  std::mutex video_mu;
  struct QueuedVideoFrame {
    rtc::binary data;
    VideoCodec codec = VideoCodec::None;
    int pt = -1;
    std::int64_t ts_ms = 0;
    bool is_keyframe = false;
  };
  std::unordered_map<int, VideoCodec> video_pt_to_codec;
  std::deque<QueuedVideoFrame> video_queue;
  rtc::binary h264_sps;
  rtc::binary h264_pps;
  rtc::binary pending_frame;
  VideoCodec pending_codec = VideoCodec::None;
  int pending_pt = -1;
  std::int64_t pending_ts_ms = 0;
  bool pending = false;
  bool pending_is_keyframe = false;
  bool vp8_waiting_for_keyframe = true;
  bool h264_waiting_for_keyframe = true;
  std::int64_t last_keyframe_request_ms = 0;
  std::int64_t last_wait_keyframe_log_ms = 0;
  std::string video_mid;
  OpenH264Decoder decoder;
  LibVpxVp8Decoder vp8_decoder;
};

WebRtcGatewayService::WebRtcGatewayService(Config cfg) : cfg_(std::move(cfg)) {
  video_force_h264_ = cfg_.video_force_h264;
  video_use_gstreamer_ = cfg_.video_use_gstreamer;
}

WebRtcGatewayService::~WebRtcGatewayService() {
  stop();
}

bool WebRtcGatewayService::restart_ws(std::string& err) {
  ws_.stop();

  WsSignalingServer::Config wcfg;
  wcfg.host = "127.0.0.1";
  wcfg.port = cfg_.ws_port;

  auto on_msg = [this](const WsSignalingServer::Event& ev) {
    enqueue_ws_event(ev);
  };
  auto on_connect = [this](const std::string& client_id, std::size_t) {
    WsSignalingServer::Event ev;
    ev.kind = WsSignalingServer::Event::Kind::Connect;
    ev.client_id = client_id;
    enqueue_ws_event(ev);
  };
  auto on_disconnect = [this](const std::string& client_id, std::size_t) {
    WsSignalingServer::Event ev;
    ev.kind = WsSignalingServer::Event::Kind::Disconnect;
    ev.client_id = client_id;
    enqueue_ws_event(ev);
  };

  return ws_.start(wcfg, std::move(on_msg), std::move(on_connect), std::move(on_disconnect), err);
}

bool WebRtcGatewayService::start() {
  if (running_.load(std::memory_order_acquire))
    return true;

  try {
    const auto lvl = rtc_level();
    // libdatachannel logs are routed through spdlog::debug; lift spdlog level when explicitly requested.
    if (lvl == rtc::LogLevel::Verbose) {
      spdlog::set_level(spdlog::level::trace);
    } else if (lvl == rtc::LogLevel::Debug) {
      spdlog::set_level(spdlog::level::debug);
    }
    if (const char* flush = std::getenv("F8_LOG_FLUSH"); flush && *flush) {
      spdlog::flush_on(spdlog::level::info);
    }
    rtc::InitLogger(lvl, [](rtc::LogLevel, rtc::string message) { spdlog::debug("rtc: {}", message); });
  } catch (...) {}

#if defined(F8_WITH_GST_WEBRTC)
  if (video_use_gstreamer_) {
    static std::once_flag gst_once;
    std::call_once(gst_once, []() { gst_init(nullptr, nullptr); });
  }
#endif

  try {
    cfg_.service_id = f8::cppsdk::ensure_token(cfg_.service_id, "service_id");
  } catch (const std::exception& e) {
    spdlog::error("invalid --service-id: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("invalid --service-id");
    return false;
  }

  if (!nats_.connect(cfg_.nats_url))
    return false;

  f8::cppsdk::KvConfig kvc;
  kvc.bucket = f8::cppsdk::kv_bucket_for_service(cfg_.service_id);
  kvc.history = 1;
  kvc.memory_storage = true;
  if (!kv_.open_or_create(nats_.jetstream(), kvc))
    return false;

  ctrl_ = std::make_unique<f8::cppsdk::ServiceControlPlaneServer>(
      f8::cppsdk::ServiceControlPlaneServer::Config{cfg_.service_id, cfg_.nats_url}, &nats_, &kv_, this);
  if (!ctrl_->start()) {
    spdlog::error("failed to start control plane");
    return false;
  }

  std::string err;
  if (!restart_ws(err)) {
    spdlog::error("failed to start websocket server: {}", err);
    return false;
  }

  // Video SHM output (decoded frames for downstream services).
  if (video_shm_name_.empty()) {
    video_shm_name_ = "shm." + cfg_.service_id + ".video";
  }
  if (!video_shm_.initialize(video_shm_name_, video_shm_bytes_, video_shm_slots_)) {
    spdlog::error("failed to initialize video shm name={} bytes={}", video_shm_name_,
                  static_cast<unsigned long long>(video_shm_bytes_));
    return false;
  }

  publish_static_state();
  publish_dynamic_state();
  f8::cppsdk::kv_set_ready(kv_, true);

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("webrtc_gateway started serviceId={} wsPort={}", cfg_.service_id, static_cast<unsigned>(cfg_.ws_port));
  return true;
}

void WebRtcGatewayService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  stop_requested_.store(true, std::memory_order_release);

  std::vector<std::unique_ptr<WebRtcSession>> to_close;
  try {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    to_close.reserve(sessions_by_id_.size());
    for (auto& kv : sessions_by_id_)
      to_close.push_back(std::move(kv.second));
    sessions_by_id_.clear();
  } catch (...) {}

  for (auto& sess : to_close) {
    if (!sess)
      continue;
    if (sess->use_gst) {
      stop_session_gst(*sess);
    }
    if (sess->pc) {
      try {
        sess->pc->close();
      } catch (...) {}
    }
  }

  try {
    ws_.stop();
  } catch (...) {}

  try {
    if (ctrl_)
      ctrl_->stop();
  } catch (...) {}
  ctrl_.reset();

  kv_.stop_watch();
  kv_.close();
  nats_.close();

  try {
    rtc::Cleanup();
  } catch (...) {}
}

void WebRtcGatewayService::enqueue_ws_event(const WsSignalingServer::Event& ev) {
  std::lock_guard<std::mutex> lock(ws_mu_);
  ws_events_.push_back(ev);
}

std::vector<WsSignalingServer::Event> WebRtcGatewayService::drain_ws_events() {
  std::vector<WsSignalingServer::Event> out;
  std::lock_guard<std::mutex> lock(ws_mu_);
  out.swap(ws_events_);
  return out;
}

void WebRtcGatewayService::enqueue_ws_send(std::string client_id, std::string text) {
  if (client_id.empty() || text.empty())
    return;
  std::lock_guard<std::mutex> lock(ws_out_mu_);
  ws_out_.push_back(WsOutbound{std::move(client_id), std::move(text)});
}

std::vector<WebRtcGatewayService::WsOutbound> WebRtcGatewayService::drain_ws_sends() {
  std::vector<WsOutbound> out;
  std::lock_guard<std::mutex> lock(ws_out_mu_);
  out.swap(ws_out_);
  return out;
}

void WebRtcGatewayService::tick() {
  if (!running_.load(std::memory_order_acquire))
    return;

  const auto events = drain_ws_events();
  if (!events.empty()) {
    const std::int64_t now = f8::cppsdk::now_ms();
    for (const auto& ev : events) {
      if (!active_.load(std::memory_order_acquire))
        continue;
      handle_ws_event(ev);
      if (ev.kind != WsSignalingServer::Event::Kind::Message || ev.text.empty())
        continue;

      json payload;
      payload["clientId"] = ev.client_id;
      payload["text"] = ev.text;
      payload["wsUrl"] = ws_url(cfg_.ws_port);
      payload["ts"] = now;

      // Try to parse as JSON so downstream can route by type/payload.
      try {
        json parsed = json::parse(ev.text, nullptr, false);
        if (parsed.is_object() || parsed.is_array()) {
          payload["json"] = parsed;
        }
      } catch (...) {}

      (void)f8::cppsdk::publish_data(nats_, cfg_.service_id, cfg_.service_id, "signalRx", payload, now);
    }
  }

  const auto out = drain_ws_sends();
  for (const auto& msg : out) {
    ws_.sendText(msg.client_id, msg.text);
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  tick_gst();
  process_video(now);
  maybe_request_periodic_keyframes(now);
  if (now - last_state_pub_ms_ >= 200) {
    publish_dynamic_state();
    last_state_pub_ms_ = now;
  }
}

#if defined(F8_WITH_GST_WEBRTC)
namespace {

struct GstCbCtx {
  WebRtcGatewayService* svc = nullptr;
  std::string client_id;
  std::string session_id;
};

static void gst_delete_cb_ctx(gpointer data, GClosure*) {
  delete static_cast<GstCbCtx*>(data);
}

struct GstAnswerCtx {
  std::string client_id;
  std::string session_id;
  std::function<GstElement*()> get_webrtc_ref;
  std::function<void(const std::string& ufrag)> set_local_ufrag;
  std::function<void(const std::string& client_id, const std::string& text)> send_ws;
};

static void gst_on_answer_created(GstPromise* promise, gpointer user_data) {
  std::unique_ptr<GstAnswerCtx> ctx(static_cast<GstAnswerCtx*>(user_data));
  if (!ctx || !ctx->get_webrtc_ref || !ctx->send_ws || !ctx->set_local_ufrag) {
    if (promise)
      gst_promise_unref(promise);
    return;
  }

  GstWebRTCSessionDescription* answer = nullptr;
  const GstStructure* reply = gst_promise_get_reply(promise);
  if (reply) {
    gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, nullptr);
  }
  gst_promise_unref(promise);

  if (!answer || !answer->sdp) {
    if (answer)
      gst_webrtc_session_description_free(answer);
    spdlog::warn("gst webrtc create-answer failed sessionId={}", ctx->session_id);
    return;
  }

  GstElement* webrtc = ctx->get_webrtc_ref();
  if (!webrtc) {
    gst_webrtc_session_description_free(answer);
    return;
  }

  g_signal_emit_by_name(webrtc, "set-local-description", answer, nullptr);

  gchar* sdp_txt = gst_sdp_message_as_text(answer->sdp);
  std::string sdp = sdp_txt ? sdp_txt : "";
  if (sdp_txt)
    g_free(sdp_txt);

  const std::string ufrag = parse_ice_ufrag(sdp);
  ctx->set_local_ufrag(ufrag);

  json out;
  out["type"] = "webrtc.answer";
  out["sessionId"] = ctx->session_id;
  out["description"] = json{{"type", "answer"}, {"sdp", sdp}};
  out["ts"] = f8::cppsdk::now_ms();
  ctx->send_ws(ctx->client_id, out.dump());
  spdlog::info("gst webrtc answer tx clientId={} sessionId={} sdpBytes={}", ctx->client_id, ctx->session_id, sdp.size());

  gst_webrtc_session_description_free(answer);
  gst_object_unref(webrtc);
}

}  // namespace
#endif

bool WebRtcGatewayService::handle_offer_gst(const std::string& client_id, const std::string& session_id,
                                           const std::string& offer_sdp) {
#if !defined(F8_WITH_GST_WEBRTC)
  (void)client_id;
  (void)session_id;
  (void)offer_sdp;
  return false;
#else
  if (!video_use_gstreamer_)
    return false;
  if (client_id.empty() || session_id.empty() || offer_sdp.empty())
    return false;

  stop_session_by_id(session_id, "replaced");

  auto session = std::make_unique<WebRtcSession>();
  session->session_id = session_id;
  session->client_id = client_id;
  session->created_ms = f8::cppsdk::now_ms();
  session->last_msg_ms = session->created_ms;
  session->use_gst = true;

  session->gst_pipeline = gst_pipeline_new(("gstpc_" + session_id).c_str());
  session->gst_webrtcbin = gst_element_factory_make("webrtcbin", "webrtc");
  if (!session->gst_pipeline || !session->gst_webrtcbin) {
    if (session->gst_pipeline)
      gst_object_unref(session->gst_pipeline);
    if (session->gst_webrtcbin)
      gst_object_unref(session->gst_webrtcbin);
    spdlog::warn("gst webrtc init failed sessionId={}", session_id);
    return false;
  }

  // Keep it simple: the gateway is LAN/local only and does not use STUN/TURN here.
  gst_bin_add(GST_BIN(session->gst_pipeline), session->gst_webrtcbin);

  g_signal_connect_data(session->gst_webrtcbin, "pad-added", G_CALLBACK(&WebRtcGatewayService::gst_on_pad_added),
                        new GstCbCtx{this, client_id, session_id}, &gst_delete_cb_ctx, static_cast<GConnectFlags>(0));
  g_signal_connect_data(session->gst_webrtcbin, "on-ice-candidate", G_CALLBACK(&WebRtcGatewayService::gst_on_ice_candidate),
                        new GstCbCtx{this, client_id, session_id}, &gst_delete_cb_ctx, static_cast<GConnectFlags>(0));

  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    sessions_by_id_[session_id] = std::move(session);
  }

  GstElement* pipeline = nullptr;
  GstElement* webrtc = nullptr;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = sessions_by_id_.find(session_id);
    if (it == sessions_by_id_.end() || !it->second)
      return false;
    pipeline = it->second->gst_pipeline;
    webrtc = it->second->gst_webrtcbin;
    if (pipeline)
      gst_object_ref(pipeline);
    if (webrtc)
      gst_object_ref(webrtc);
  }

  if (!pipeline || !webrtc) {
    if (pipeline)
      gst_object_unref(pipeline);
    if (webrtc)
      gst_object_unref(webrtc);
    return false;
  }

  (void)gst_element_set_state(pipeline, GST_STATE_PLAYING);

  GstSDPMessage* sdp = nullptr;
  if (gst_sdp_message_new(&sdp) != GST_SDP_OK || !sdp) {
    spdlog::warn("gst webrtc sdp alloc failed sessionId={}", session_id);
    gst_object_unref(pipeline);
    gst_object_unref(webrtc);
    stop_session_by_id(session_id, "gst_sdp_alloc_failed");
    return false;
  }
  const GstSDPResult parse_res =
      gst_sdp_message_parse_buffer(reinterpret_cast<const guint8*>(offer_sdp.data()), offer_sdp.size(), sdp);
  if (parse_res != GST_SDP_OK) {
    spdlog::warn("gst webrtc sdp parse failed sessionId={} err={}", session_id, static_cast<int>(parse_res));
    gst_sdp_message_free(sdp);
    gst_object_unref(pipeline);
    gst_object_unref(webrtc);
    stop_session_by_id(session_id, "gst_sdp_parse_failed");
    return false;
  }

  GstWebRTCSessionDescription* offer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
  if (!offer) {
    gst_sdp_message_free(sdp);
    gst_object_unref(pipeline);
    gst_object_unref(webrtc);
    stop_session_by_id(session_id, "gst_offer_failed");
    return false;
  }

  g_signal_emit_by_name(webrtc, "set-remote-description", offer, nullptr);
  gst_webrtc_session_description_free(offer);

  auto* ans_ctx = new GstAnswerCtx{};
  ans_ctx->client_id = client_id;
  ans_ctx->session_id = session_id;
  ans_ctx->get_webrtc_ref = [this, session_id]() -> GstElement* {
    GstElement* w = nullptr;
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = sessions_by_id_.find(session_id);
    if (it != sessions_by_id_.end() && it->second) {
      w = it->second->gst_webrtcbin;
      if (w)
        gst_object_ref(w);
    }
    return w;
  };
  ans_ctx->set_local_ufrag = [this, session_id](const std::string& ufrag) {
    if (ufrag.empty())
      return;
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = sessions_by_id_.find(session_id);
    if (it != sessions_by_id_.end() && it->second) {
      it->second->local_ice_ufrag = ufrag;
      it->second->answer_sent = true;
    }
  };
  ans_ctx->send_ws = [this](const std::string& cid, const std::string& text) { enqueue_ws_send(cid, text); };

  GstPromise* answer_promise = gst_promise_new_with_change_func(&gst_on_answer_created, ans_ctx, nullptr);
  g_signal_emit_by_name(webrtc, "create-answer", nullptr, answer_promise);

  spdlog::info("gst webrtc offer accepted clientId={} sessionId={}", client_id, session_id);
  gst_object_unref(pipeline);
  gst_object_unref(webrtc);
  return true;
#endif
}

bool WebRtcGatewayService::handle_ice_gst(const std::string& session_id, int mline, const std::string& candidate) {
#if !defined(F8_WITH_GST_WEBRTC)
  (void)session_id;
  (void)mline;
  (void)candidate;
  return false;
#else
  if (!video_use_gstreamer_)
    return false;
  GstElement* webrtc = nullptr;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = sessions_by_id_.find(session_id);
    if (it == sessions_by_id_.end() || !it->second || !it->second->use_gst)
      return false;
    webrtc = it->second->gst_webrtcbin;
    if (webrtc)
      gst_object_ref(webrtc);
    it->second->last_msg_ms = f8::cppsdk::now_ms();
  }
  if (!webrtc)
    return false;
  g_signal_emit_by_name(webrtc, "add-ice-candidate", static_cast<guint>(std::max(0, mline)), candidate.c_str());
  gst_object_unref(webrtc);
  ice_rx_.fetch_add(1, std::memory_order_relaxed);
  return true;
#endif
}

void WebRtcGatewayService::stop_session_gst(WebRtcSession& session) {
#if defined(F8_WITH_GST_WEBRTC)
  if (session.gst_pipeline) {
    (void)gst_element_set_state(session.gst_pipeline, GST_STATE_NULL);
    gst_object_unref(session.gst_pipeline);
  }
  session.gst_pipeline = nullptr;
  session.gst_webrtcbin = nullptr;
  session.gst_appsink = nullptr;
  session.gst_video_branch_linked = false;
#else
  (void)session;
#endif
}

void WebRtcGatewayService::tick_gst() {
#if !defined(F8_WITH_GST_WEBRTC)
  return;
#else
  if (!video_use_gstreamer_)
    return;

  std::vector<std::string> to_stop;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    for (const auto& kv : sessions_by_id_) {
      const auto& sess = kv.second;
      if (!sess || !sess->use_gst || !sess->gst_pipeline)
        continue;
      GstBus* bus = gst_element_get_bus(sess->gst_pipeline);
      if (!bus)
        continue;
      for (;;) {
        GstMessage* msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (!msg)
          break;
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
          spdlog::info("gst webrtc eos sessionId={}", sess->session_id);
          to_stop.push_back(sess->session_id);
        } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
          GError* err = nullptr;
          gchar* dbg = nullptr;
          gst_message_parse_error(msg, &err, &dbg);
          spdlog::warn("gst webrtc error sessionId={} err={} dbg={}", sess->session_id, err ? err->message : "",
                       dbg ? dbg : "");
          if (err)
            g_error_free(err);
          if (dbg)
            g_free(dbg);
          to_stop.push_back(sess->session_id);
        }
        gst_message_unref(msg);
      }
      gst_object_unref(bus);
    }
  }

  for (const auto& sid : to_stop) {
    stop_session_by_id(sid, "gst_bus_error");
  }
#endif
}

void WebRtcGatewayService::gst_on_pad_added(void* /*webrtc*/, void* pad, void* user_data) {
#if !defined(F8_WITH_GST_WEBRTC)
  (void)pad;
  (void)user_data;
#else
  auto* ctx = static_cast<GstCbCtx*>(user_data);
  auto* new_pad = static_cast<GstPad*>(pad);
  if (!ctx || !ctx->svc || !new_pad)
    return;

  GstElement* pipeline = nullptr;
  bool already_linked = false;
  {
    std::lock_guard<std::mutex> lock(ctx->svc->rtc_mu_);
    auto it = ctx->svc->sessions_by_id_.find(ctx->session_id);
    if (it == ctx->svc->sessions_by_id_.end() || !it->second || !it->second->use_gst)
      return;
    pipeline = it->second->gst_pipeline;
    already_linked = it->second->gst_video_branch_linked;
    if (pipeline)
      gst_object_ref(pipeline);
  }
  if (!pipeline || already_linked) {
    if (pipeline)
      gst_object_unref(pipeline);
    return;
  }

  GstCaps* caps = gst_pad_get_current_caps(new_pad);
  if (!caps) {
    caps = gst_pad_query_caps(new_pad, nullptr);
  }
  if (!caps || gst_caps_is_empty(caps)) {
    if (caps)
      gst_caps_unref(caps);
    gst_object_unref(pipeline);
    return;
  }

  const GstStructure* s = gst_caps_get_structure(caps, 0);
  const char* media = s ? gst_structure_get_string(s, "media") : nullptr;
  const char* encoding = s ? gst_structure_get_string(s, "encoding-name") : nullptr;
  if (!media || std::string(media) != "video" || !encoding) {
    gst_caps_unref(caps);
    gst_object_unref(pipeline);
    return;
  }

  std::string enc = encoding;
  std::transform(enc.begin(), enc.end(), enc.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

  GstElement* queue = gst_element_factory_make("queue", nullptr);
  GstElement* depay = nullptr;
  GstElement* parser = nullptr;
  GstElement* decoder = nullptr;
  if (enc == "VP8") {
    depay = gst_element_factory_make("rtpvp8depay", nullptr);
    decoder = gst_element_factory_make("vp8dec", nullptr);
  } else if (enc == "H264") {
    depay = gst_element_factory_make("rtph264depay", nullptr);
    parser = gst_element_factory_make("h264parse", nullptr);
    decoder = gst_element_factory_make("openh264dec", nullptr);
  }

  GstElement* convert = gst_element_factory_make("videoconvert", nullptr);
  GstElement* capsfilter = gst_element_factory_make("capsfilter", nullptr);
  GstElement* appsink = gst_element_factory_make("appsink", nullptr);

  if (!queue || !depay || !decoder || !convert || !capsfilter || !appsink) {
    if (queue)
      gst_object_unref(queue);
    if (depay)
      gst_object_unref(depay);
    if (parser)
      gst_object_unref(parser);
    if (decoder)
      gst_object_unref(decoder);
    if (convert)
      gst_object_unref(convert);
    if (capsfilter)
      gst_object_unref(capsfilter);
    if (appsink)
      gst_object_unref(appsink);
    gst_caps_unref(caps);
    gst_object_unref(pipeline);
    return;
  }

  GstCaps* out_caps = gst_caps_from_string("video/x-raw,format=BGRA");
  g_object_set(capsfilter, "caps", out_caps, nullptr);
  gst_caps_unref(out_caps);

  g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, "max-buffers", 1, "drop", TRUE, nullptr);
  g_signal_connect_data(appsink, "new-sample", G_CALLBACK(&WebRtcGatewayService::gst_on_new_sample),
                        new GstCbCtx{ctx->svc, ctx->client_id, ctx->session_id}, &gst_delete_cb_ctx,
                        static_cast<GConnectFlags>(0));

  if (parser) {
    gst_bin_add_many(GST_BIN(pipeline), queue, depay, parser, decoder, convert, capsfilter, appsink, nullptr);
  } else {
    gst_bin_add_many(GST_BIN(pipeline), queue, depay, decoder, convert, capsfilter, appsink, nullptr);
  }

  const gboolean link_ok = parser
                               ? gst_element_link_many(queue, depay, parser, decoder, convert, capsfilter, appsink, nullptr)
                               : gst_element_link_many(queue, depay, decoder, convert, capsfilter, appsink, nullptr);
  if (!link_ok) {
    spdlog::warn("gst video branch link failed sessionId={} codec={}", ctx->session_id, enc);
    gst_caps_unref(caps);
    gst_object_unref(pipeline);
    return;
  }

  GstPad* sink_pad = gst_element_get_static_pad(queue, "sink");
  const auto pad_link_res = sink_pad ? gst_pad_link(new_pad, sink_pad) : GST_PAD_LINK_REFUSED;
  if (sink_pad)
    gst_object_unref(sink_pad);
  gst_caps_unref(caps);
  if (pad_link_res != GST_PAD_LINK_OK) {
    spdlog::warn("gst pad link failed sessionId={} res={}", ctx->session_id, static_cast<int>(pad_link_res));
    gst_object_unref(pipeline);
    return;
  }

  gst_element_sync_state_with_parent(queue);
  gst_element_sync_state_with_parent(depay);
  if (parser)
    gst_element_sync_state_with_parent(parser);
  gst_element_sync_state_with_parent(decoder);
  gst_element_sync_state_with_parent(convert);
  gst_element_sync_state_with_parent(capsfilter);
  gst_element_sync_state_with_parent(appsink);

  {
    std::lock_guard<std::mutex> lock(ctx->svc->rtc_mu_);
    auto it = ctx->svc->sessions_by_id_.find(ctx->session_id);
    if (it != ctx->svc->sessions_by_id_.end() && it->second && it->second->use_gst) {
      it->second->gst_appsink = appsink;
      it->second->gst_video_branch_linked = true;
    }
  }

  spdlog::info("gst video pad linked sessionId={} codec={}", ctx->session_id, enc);
  gst_object_unref(pipeline);
#endif
}

void WebRtcGatewayService::gst_on_ice_candidate(void* /*webrtc*/, unsigned mline, char* candidate, void* user_data) {
#if !defined(F8_WITH_GST_WEBRTC)
  (void)mline;
  (void)candidate;
  (void)user_data;
#else
  auto* ctx = static_cast<GstCbCtx*>(user_data);
  if (!ctx || !ctx->svc || !candidate || !*candidate)
    return;

  std::string ufrag;
  {
    std::lock_guard<std::mutex> lock(ctx->svc->rtc_mu_);
    auto it = ctx->svc->sessions_by_id_.find(ctx->session_id);
    if (it != ctx->svc->sessions_by_id_.end() && it->second) {
      ufrag = it->second->local_ice_ufrag;
    }
  }

  json out;
  out["type"] = "webrtc.ice";
  out["sessionId"] = ctx->session_id;
  out["candidate"] =
      json{{"candidate", std::string(candidate)}, {"sdpMid", std::to_string(mline)}, {"sdpMLineIndex", static_cast<int>(mline)}};
  if (!ufrag.empty()) {
    out["candidate"]["usernameFragment"] = ufrag;
  }
  out["ts"] = f8::cppsdk::now_ms();
  ctx->svc->enqueue_ws_send(ctx->client_id, out.dump());
  ctx->svc->ice_tx_.fetch_add(1, std::memory_order_relaxed);
#endif
}

int WebRtcGatewayService::gst_on_new_sample(void* appsink, void* user_data) {
#if !defined(F8_WITH_GST_WEBRTC)
  (void)appsink;
  (void)user_data;
  return 0;
#else
  auto* ctx = static_cast<GstCbCtx*>(user_data);
  if (!ctx || !ctx->svc || !appsink)
    return static_cast<int>(GST_FLOW_OK);

  GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
  if (!sample)
    return static_cast<int>(GST_FLOW_OK);

  GstCaps* caps = gst_sample_get_caps(sample);
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!caps || !buffer) {
    gst_sample_unref(sample);
    return static_cast<int>(GST_FLOW_OK);
  }

  GstVideoInfo info;
  if (!gst_video_info_from_caps(&info, caps)) {
    gst_sample_unref(sample);
    return static_cast<int>(GST_FLOW_OK);
  }

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ) || !map.data || map.size == 0) {
    gst_sample_unref(sample);
    return static_cast<int>(GST_FLOW_OK);
  }

  const unsigned width = static_cast<unsigned>(GST_VIDEO_INFO_WIDTH(&info));
  const unsigned height = static_cast<unsigned>(GST_VIDEO_INFO_HEIGHT(&info));
  const unsigned stride = static_cast<unsigned>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
  const std::int64_t now_ms = f8::cppsdk::now_ms();

  ctx->svc->video_frames_rx_.fetch_add(1, std::memory_order_relaxed);
  ctx->svc->video_last_frame_bytes_.store(static_cast<std::uint64_t>(map.size), std::memory_order_relaxed);
  ctx->svc->video_last_frame_ts_ms_.store(now_ms, std::memory_order_relaxed);

  bool wrote = false;
  {
    std::lock_guard<std::mutex> lock(ctx->svc->video_mu_);
    if (!ctx->svc->video_enabled_) {
      // Drop frame.
    } else {
      const int max_fps = ctx->svc->video_shm_max_fps_;
      if (max_fps > 0) {
        const std::int64_t min_dt = 1000 / std::max(1, max_fps);
        if (now_ms - ctx->svc->last_video_write_ms_ < min_dt) {
          // Drop frame (throttle).
        } else {
          (void)ctx->svc->video_shm_.ensureConfiguration(width, height);
          wrote = ctx->svc->video_shm_.writeFrame(map.data, stride);
          ctx->svc->last_video_write_ms_ = now_ms;
        }
      } else {
        (void)ctx->svc->video_shm_.ensureConfiguration(width, height);
        wrote = ctx->svc->video_shm_.writeFrame(map.data, stride);
        ctx->svc->last_video_write_ms_ = now_ms;
      }
    }
  }

  if (wrote) {
    ctx->svc->video_frames_decoded_.fetch_add(1, std::memory_order_relaxed);
    ctx->svc->video_frames_written_.fetch_add(1, std::memory_order_relaxed);
  }

  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);
  return static_cast<int>(GST_FLOW_OK);
#endif
}

void WebRtcGatewayService::maybe_request_periodic_keyframes(std::int64_t now_ms) {
  if (!video_enabled_)
    return;
  if (!active_.load(std::memory_order_acquire))
    return;

  static constexpr std::int64_t kIntervalMs = 2000;

  std::vector<std::shared_ptr<rtc::Track>> tracks;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    tracks.reserve(sessions_by_id_.size());
    for (auto& kv : sessions_by_id_) {
      auto* sess = kv.second.get();
      if (!sess)
        continue;
      std::lock_guard<std::mutex> lk(sess->video_mu);
      if (!sess->remote_video)
        continue;
      if (sess->last_keyframe_request_ms != 0 && (now_ms - sess->last_keyframe_request_ms) < kIntervalMs)
        continue;
      sess->last_keyframe_request_ms = now_ms;
      tracks.push_back(sess->remote_video);
    }
  }

  for (auto& tr : tracks) {
    try {
      (void)tr->requestKeyframe();
    } catch (...) {
    }
  }
}

void WebRtcGatewayService::process_video(std::int64_t now_ms) {
  if (!video_enabled_)
    return;
  if (!active_.load(std::memory_order_acquire))
    return;

  const int max_fps = video_shm_max_fps_;
  if (max_fps > 0) {
    const std::int64_t min_dt = std::max<std::int64_t>(1, 1000 / static_cast<std::int64_t>(max_fps));
    if (last_video_write_ms_ != 0 && (now_ms - last_video_write_ms_) < min_dt) {
      return;
    }
  }

  std::string best_sid;
  WebRtcSession* best_sess = nullptr;
  std::int64_t best_ts = 0;

  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    for (auto& kv : sessions_by_id_) {
      auto* sess = kv.second.get();
      if (!sess)
        continue;
      std::lock_guard<std::mutex> lk(sess->video_mu);
      if (sess->video_queue.empty())
        continue;
      const auto ts = sess->video_queue.back().ts_ms;
      if (ts >= best_ts) {
        best_ts = ts;
        best_sid = kv.first;
        best_sess = sess;
      }
    }
  }

  if (!best_sess)
    return;

  rtc::binary frame;
  VideoCodec codec = VideoCodec::None;
  bool frame_is_keyframe = false;
  int frame_pt = -1;
  std::string mid;
  rtc::binary h264_sps;
  rtc::binary h264_pps;
  {
    std::lock_guard<std::mutex> lk(best_sess->video_mu);
    if (best_sess->video_queue.empty()) {
      return;
    }

    // Keep decoding in order to preserve inter-frame references (H264/VP8 P-frames depend on previous frames).
    // If backlog grows, drop everything up to the last keyframe and restart from there.
    static constexpr std::size_t kMaxBacklog = 90;
    if (best_sess->video_queue.size() > kMaxBacklog) {
      std::optional<std::size_t> last_kf;
      for (std::size_t i = best_sess->video_queue.size(); i-- > 0;) {
        if (best_sess->video_queue[i].is_keyframe) {
          last_kf = i;
          break;
        }
      }
      if (last_kf.has_value() && *last_kf > 0) {
        spdlog::warn("webrtc video backlog drop sessionId={} dropFrames={} keepFrom={} queueBefore={}", best_sid, *last_kf,
                     *last_kf, best_sess->video_queue.size());
        for (std::size_t i = 0; i < *last_kf; ++i) {
          best_sess->video_queue.pop_front();
        }
        // Restart decoder on the keyframe boundary.
        if (!best_sess->video_queue.empty()) {
          const auto c = best_sess->video_queue.front().codec;
          if (c == VideoCodec::VP8) {
            best_sess->vp8_decoder.reset();
            best_sess->vp8_waiting_for_keyframe = false;
          } else if (c == VideoCodec::H264) {
            best_sess->decoder.reset();
            best_sess->h264_waiting_for_keyframe = false;
          }
        }
      } else if (!best_sess->video_queue.empty()) {
        spdlog::warn("webrtc video backlog drop sessionId={} no_keyframe queueBefore={} codec={}", best_sid,
                     best_sess->video_queue.size(), static_cast<int>(best_sess->video_queue.back().codec));
        // No keyframe in backlog: drop and wait for the next keyframe.
        const auto c = best_sess->video_queue.back().codec;
        best_sess->video_queue.clear();
        if (c == VideoCodec::VP8) {
          best_sess->vp8_decoder.reset();
          best_sess->vp8_waiting_for_keyframe = true;
        } else if (c == VideoCodec::H264) {
          best_sess->decoder.reset();
          best_sess->h264_waiting_for_keyframe = true;
        }
        return;
      }
    }

    auto q = std::move(best_sess->video_queue.front());
    best_sess->video_queue.pop_front();
    frame = std::move(q.data);
    codec = q.codec;
    frame_pt = q.pt;
    frame_is_keyframe = q.is_keyframe;
    mid = best_sess->video_mid;
    h264_sps = best_sess->h264_sps;
    h264_pps = best_sess->h264_pps;
  }

  if (frame.empty() || codec == VideoCodec::None)
    return;

  auto maybe_request_keyframe = [&](const char* reason) {
    std::shared_ptr<rtc::Track> tr;
    {
      std::lock_guard<std::mutex> lk(best_sess->video_mu);
      if (!best_sess->remote_video) {
        return;
      }
      const std::int64_t last = best_sess->last_keyframe_request_ms;
      if (last != 0 && (now_ms - last) < 500) {
        return;
      }
      best_sess->last_keyframe_request_ms = now_ms;
      tr = best_sess->remote_video;
    }
    try {
      (void)tr->requestKeyframe();
    } catch (...) {
    }
    spdlog::debug("webrtc request keyframe sessionId={} reason={}", best_sid, reason);
  };

  cv::Mat bgra;
  unsigned src_w = 0;
  unsigned src_h = 0;
  std::string derr;
  const std::int64_t t0 = f8::cppsdk::now_ms();
  bool ok = false;
  std::string codec_name = "unknown";
  if (codec == VideoCodec::H264) {
    codec_name = "H264";
    rtc::binary annexb;
    if (OpenH264Decoder::looks_like_annexb(frame)) {
      annexb = frame;
    } else {
      (void)OpenH264Decoder::avcc_to_annexb(frame, annexb);
    }

    // Opportunistically learn SPS/PPS from the bitstream.
    if (!annexb.empty()) {
      rtc::binary sps_found;
      rtc::binary pps_found;
      h264_cache_param_sets_from_annexb(annexb, sps_found, pps_found);
      if (!sps_found.empty() || !pps_found.empty()) {
        std::lock_guard<std::mutex> lk(best_sess->video_mu);
        if (!sps_found.empty()) {
          best_sess->h264_sps = sps_found;
          h264_sps = sps_found;
        }
        if (!pps_found.empty()) {
          best_sess->h264_pps = pps_found;
          h264_pps = pps_found;
        }
      }
    }

    // If the keyframe doesn't carry SPS/PPS, prepend cached SDP/inband parameter sets.
    if (frame_is_keyframe && !h264_sps.empty() && !h264_pps.empty()) {
      rtc::binary sps_found;
      rtc::binary pps_found;
      if (!annexb.empty()) {
        h264_cache_param_sets_from_annexb(annexb, sps_found, pps_found);
      }
      if (sps_found.empty() || pps_found.empty()) {
        rtc::binary merged;
        merged.reserve(h264_sps.size() + h264_pps.size() + annexb.size());
        merged.insert(merged.end(), h264_sps.begin(), h264_sps.end());
        merged.insert(merged.end(), h264_pps.begin(), h264_pps.end());
        merged.insert(merged.end(), annexb.begin(), annexb.end());
        annexb = std::move(merged);
      }
    }

    int drv = 0;
    ok = best_sess->decoder.decode_bgra(annexb.empty() ? frame : annexb, bgra, video_shm_max_width_, video_shm_max_height_, src_w,
                                        src_h, derr, drv);
    if (!ok && drv != 0) {
      derr += " rv=" + std::to_string(drv);
    }
  } else if (codec == VideoCodec::VP8) {
    codec_name = "VP8";
    ok = best_sess->vp8_decoder.decode_bgra(frame, bgra, video_shm_max_width_, video_shm_max_height_, src_w, src_h, derr);
  }
  const std::int64_t t1 = f8::cppsdk::now_ms();
  video_last_decode_ms_.store(std::max<std::int64_t>(0, t1 - t0), std::memory_order_relaxed);
  if (!ok) {
    if (codec == VideoCodec::VP8) {
      const bool is_keyframe =
          frame_is_keyframe || (!frame.empty() && ((std::to_integer<std::uint8_t>(frame[0]) & 0x01u) == 0));
      if (derr == "vp8_waiting_for_keyframe") {
        {
          std::lock_guard<std::mutex> lk(best_sess->video_mu);
          best_sess->vp8_waiting_for_keyframe = true;
          if (best_sess->last_wait_keyframe_log_ms == 0 || (now_ms - best_sess->last_wait_keyframe_log_ms) >= 1000) {
            best_sess->last_wait_keyframe_log_ms = now_ms;
            spdlog::info("webrtc video waiting keyframe sessionId={} pt={} lastFrameBytes={}", best_sid,
                         video_last_rtp_pt_.load(std::memory_order_relaxed), frame.size());
          }
        }
        maybe_request_keyframe("vp8_waiting_keyframe");
        std::lock_guard<std::mutex> lk(video_err_mu_);
        video_last_error_ = derr;
        return;
      }
      if (!is_keyframe && derr.find("Bitstream not supported") != std::string::npos) {
        best_sess->vp8_decoder.reset();
        {
          std::lock_guard<std::mutex> lk(best_sess->video_mu);
          best_sess->vp8_waiting_for_keyframe = true;
        }
        maybe_request_keyframe("vp8_unsup_bitstream");
      }
    } else if (codec == VideoCodec::H264) {
      {
        std::lock_guard<std::mutex> lk(best_sess->video_mu);
        best_sess->h264_waiting_for_keyframe = true;
        best_sess->video_queue.clear();
      }
      maybe_request_keyframe("h264_decode_failed");
    }

    video_decode_errors_.fetch_add(1, std::memory_order_relaxed);
    const auto last_pt = (frame_pt >= 0) ? frame_pt : video_last_rtp_pt_.load(std::memory_order_relaxed);
    spdlog::warn("webrtc video decode failed sessionId={} codec={} pt={} frameBytes={} head={} err={}", best_sid, codec_name,
                 last_pt, frame.size(), hex_prefix(frame, 12), derr);
    std::lock_guard<std::mutex> lk(video_err_mu_);
    video_last_error_ = derr;
    return;
  }
  if (codec == VideoCodec::VP8) {
    std::lock_guard<std::mutex> lk(best_sess->video_mu);
    best_sess->vp8_waiting_for_keyframe = false;
  } else if (codec == VideoCodec::H264) {
    std::lock_guard<std::mutex> lk(best_sess->video_mu);
    best_sess->h264_waiting_for_keyframe = false;
  }
  video_frames_decoded_.fetch_add(1, std::memory_order_relaxed);

  const unsigned dst_w = static_cast<unsigned>(bgra.cols);
  const unsigned dst_h = static_cast<unsigned>(bgra.rows);
  if (dst_w == 0 || dst_h == 0)
    return;

  {
    std::lock_guard<std::mutex> lock(video_mu_);
    if (!video_shm_.ensureConfiguration(dst_w, dst_h)) {
      return;
    }
    const unsigned shm_w = video_shm_.outputWidth();
    const unsigned shm_h = video_shm_.outputHeight();
    if (shm_w != dst_w || shm_h != dst_h) {
      cv::Mat resized;
      cv::resize(bgra, resized, cv::Size(static_cast<int>(shm_w), static_cast<int>(shm_h)), 0.0, 0.0, cv::INTER_LINEAR);
      if (!resized.isContinuous()) {
        resized = resized.clone();
      }
      if (video_shm_.writeFrame(resized.data, static_cast<unsigned>(resized.step))) {
        video_frames_written_.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      if (video_shm_.writeFrame(bgra.data, static_cast<unsigned>(bgra.step))) {
        video_frames_written_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  last_video_write_ms_ = now_ms;

  // Publish stable metadata updates (dimensions, SHM names, etc) only when changed.
  const json meta = json{{"via", "video"}};
  std::vector<std::pair<std::string, json>> updates;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto want = [&](const std::string& field, const json& v) {
      auto it = published_state_.find(field);
      if (it != published_state_.end() && it->second == v)
        return;
      published_state_[field] = v;
      updates.emplace_back(field, v);
    };
    want("videoShmName", video_shm_.regionName());
    want("videoShmEvent", video_shm_.frameEventName());
    want("videoCodec", codec_name);
    want("videoSrcWidth", static_cast<int>(src_w));
    want("videoSrcHeight", static_cast<int>(src_h));
    want("videoWidth", static_cast<int>(video_shm_.outputWidth()));
    want("videoHeight", static_cast<int>(video_shm_.outputHeight()));
    want("videoPitch", static_cast<int>(video_shm_.outputPitch()));
    want("videoSessionId", best_sid);
    if (!mid.empty()) {
      want("videoMid", mid);
    }
  }
  for (const auto& [field, v] : updates) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
  }
}

void WebRtcGatewayService::handle_ws_event(const WsSignalingServer::Event& ev) {
  if (ev.kind == WsSignalingServer::Event::Kind::Connect) {
    spdlog::info("ws connect clientId={} connections={}", ev.client_id, ws_.connectionCount());
    return;
  }
  if (ev.kind == WsSignalingServer::Event::Kind::Disconnect) {
    spdlog::info("ws disconnect clientId={} connections={}", ev.client_id, ws_.connectionCount());
    stop_sessions_by_client(ev.client_id, "ws_disconnect");
    return;
  }

  if (ev.text.empty())
    return;
  try {
    json parsed = json::parse(ev.text, nullptr, false);
    if (parsed.is_object())
      handle_ws_json_message(ev, parsed);
  } catch (...) {}
}

void WebRtcGatewayService::handle_ws_json_message(const WsSignalingServer::Event& ev, const nlohmann::json& msg) {
  const auto type = json_string(msg, "type").value_or("");
  if (type.empty())
    return;

  if (type == "hello") {
    spdlog::info("ws hello clientId={} msg={}", ev.client_id, msg.dump());
    return;
  }

  if (type == "webrtc.stop") {
    const auto sid = json_string(msg, "sessionId").value_or("");
    if (!sid.empty())
      stop_session_by_id(sid, json_string(msg, "reason").value_or("stopped"));
    return;
  }

  if (type == "webrtc.ice") {
    const auto sid = json_string(msg, "sessionId").value_or("");
    if (sid.empty())
      return;
    const auto candObj = json_object(msg, "candidate");
    if (!candObj)
      return;
    const auto candStr = json_string(*candObj, "candidate").value_or("");
    if (candStr.empty())
      return;
    const auto mid = json_string(*candObj, "sdpMid").value_or("");
    int mline = 0;
    try {
      if (candObj->contains("sdpMLineIndex") && (*candObj)["sdpMLineIndex"].is_number_integer()) {
        mline = (*candObj)["sdpMLineIndex"].get<int>();
      } else if (candObj->contains("sdpMLineIndex") && (*candObj)["sdpMLineIndex"].is_number()) {
        mline = static_cast<int>((*candObj)["sdpMLineIndex"].get<double>());
      }
    } catch (...) {
      mline = 0;
    }

    bool is_gst = false;
    {
      std::lock_guard<std::mutex> lock(rtc_mu_);
      auto it = sessions_by_id_.find(sid);
      if (it == sessions_by_id_.end() || !it->second)
        return;
      it->second->last_msg_ms = f8::cppsdk::now_ms();
      is_gst = it->second->use_gst;
      if (!is_gst && !it->second->pc) {
        return;
      }
    }
    if (is_gst) {
      (void)handle_ice_gst(sid, mline, candStr);
      return;
    }

    rtc::PeerConnection* pc = nullptr;
    std::string mid_final = mid;
    {
      std::lock_guard<std::mutex> lock(rtc_mu_);
      auto it = sessions_by_id_.find(sid);
      if (it == sessions_by_id_.end() || !it->second || !it->second->pc)
        return;
      pc = it->second->pc.get();
    }

    try {
      if (mid_final.empty() && mline > 0) {
        mid_final = std::to_string(mline);
      }
      rtc::Candidate c(mid_final.empty() ? rtc::Candidate(candStr) : rtc::Candidate(candStr, mid_final));
      try {
        (void)c.resolve(rtc::Candidate::ResolveMode::Lookup);
      } catch (...) {
      }
      pc->addRemoteCandidate(c);
      ice_rx_.fetch_add(1, std::memory_order_relaxed);
      const auto addr = c.address().value_or("");
      const auto port = c.port().has_value() ? std::to_string(*c.port()) : "";
      spdlog::info("webrtc ice rx clientId={} sessionId={} mid={} mline={} type={} addr={}:{} candBytes={}", ev.client_id,
                   sid, mid_final, mline, static_cast<int>(c.type()), addr, port, candStr.size());
    } catch (const std::exception& e) {
      spdlog::warn("webrtc ice rx failed sessionId={} err={}", sid, e.what());
      std::lock_guard<std::mutex> lk(ice_err_mu_);
      ice_last_error_ = std::string("ice_rx_failed: ") + e.what();
    } catch (...) {
      spdlog::warn("webrtc ice rx failed sessionId={}", sid);
      std::lock_guard<std::mutex> lk(ice_err_mu_);
      ice_last_error_ = "ice_rx_failed";
    }
    return;
  }

  if (type == "webrtc.offer") {
    const auto sid = json_string(msg, "sessionId").value_or("");
    const auto descObj = json_object(msg, "description");
    if (sid.empty() || !descObj)
      return;

    const auto sdpType = json_string(*descObj, "type").value_or("offer");
    const auto sdp = json_string(*descObj, "sdp").value_or("");
    if (sdp.empty())
      return;

    spdlog::info("webrtc offer rx clientId={} sessionId={} sdpBytes={}", ev.client_id, sid, sdp.size());
    spdlog::info("webrtc remote description summary sessionId={} sdpSummary={}", sid, summarize_sdp_mlines(sdp));

    if (video_use_gstreamer_) {
      if (handle_offer_gst(ev.client_id, sid, sdp)) {
        return;
      }
      spdlog::warn("webrtc gst offer handling failed; falling back to libdatachannel sessionId={}", sid);
    }

    auto session = std::make_unique<WebRtcSession>();
    session->session_id = sid;
    session->client_id = ev.client_id;
    session->created_ms = f8::cppsdk::now_ms();
    session->last_msg_ms = session->created_ms;

    rtc::Configuration cfg;
    // Localhost-only: prefer host candidates, avoid STUN/srflx routes (more failure modes and higher latency).
    cfg.forceMediaTransport = true;
    // We want to set track descriptions before generating the answer. With auto negotiation enabled,
    // libdatachannel may auto-generate an answer immediately after setRemoteDescription(), which has been producing
    // rejected m-lines (m=... 0). We'll generate the answer explicitly after we receive tracks.
    cfg.disableAutoNegotiation = true;
    session->pc = std::make_unique<rtc::PeerConnection>(cfg);
    rtc::PeerConnection* const pc = session->pc.get();

    const std::string clientId = ev.client_id;
    const std::string sessionId = sid;

    session->pc->onStateChange([clientId, sessionId](rtc::PeerConnection::State state) {
      spdlog::info("webrtc pc state clientId={} sessionId={} state={}({})", clientId, sessionId,
                   static_cast<int>(state), pc_state_name(state));
    });
    session->pc->onIceStateChange([clientId, sessionId](rtc::PeerConnection::IceState state) {
      spdlog::info("webrtc ice state clientId={} sessionId={} state={}({})", clientId, sessionId,
                   static_cast<int>(state), ice_state_name(state));
    });
    session->pc->onGatheringStateChange([clientId, sessionId](rtc::PeerConnection::GatheringState state) {
      spdlog::info("webrtc gather state clientId={} sessionId={} state={}({})", clientId, sessionId,
                   static_cast<int>(state), gathering_state_name(state));
    });
    session->pc->onSignalingStateChange([clientId, sessionId](rtc::PeerConnection::SignalingState state) {
      spdlog::info("webrtc signaling state clientId={} sessionId={} state={}({})", clientId, sessionId,
                   static_cast<int>(state), signaling_state_name(state));
    });

    session->pc->onLocalDescription([this, clientId, sessionId](rtc::Description desc) {
      const std::string sdp = desc.generateSdp();
      const std::string ufrag = parse_ice_ufrag(sdp);
      spdlog::info("webrtc local description clientId={} sessionId={} type={} sdpSummary={}", clientId, sessionId,
                   desc.typeString(), summarize_sdp_mlines(sdp));
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second) {
          if (!ufrag.empty()) {
            it->second->local_ice_ufrag = ufrag;
          }
          if (it->second->answer_sent)
            return;
          it->second->answer_sent = true;
        }
      }
      json out;
      out["type"] = "webrtc.answer";
      out["sessionId"] = sessionId;
      out["description"] = json{{"type", desc.typeString()}, {"sdp", sdp}};
      out["ts"] = f8::cppsdk::now_ms();
      enqueue_ws_send(clientId, out.dump());
      spdlog::info("webrtc answer tx clientId={} sessionId={} sdpBytes={}", clientId, sessionId,
                   out["description"]["sdp"].get<std::string>().size());
    });

    session->pc->onLocalCandidate([this, clientId, sessionId](rtc::Candidate cand) {
      const auto cand_str = cand.candidate();
      if (cand_str.empty()) {
        spdlog::info("webrtc ice tx done clientId={} sessionId={}", clientId, sessionId);
        return;
      }
      ice_tx_.fetch_add(1, std::memory_order_relaxed);
      std::string ufrag;
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second) {
          ufrag = it->second->local_ice_ufrag;
        }
      }
      json out;
      out["type"] = "webrtc.ice";
      out["sessionId"] = sessionId;
      const std::string mid = cand.mid();
      out["candidate"] = json{{"candidate", cand_str}, {"sdpMid", mid}, {"sdpMLineIndex", mid_to_mline_index(mid)}};
      if (!ufrag.empty()) {
        out["candidate"]["usernameFragment"] = ufrag;
      }
      out["ts"] = f8::cppsdk::now_ms();
      enqueue_ws_send(clientId, out.dump());
      const auto addr = cand.address().value_or("");
      const auto port = cand.port().has_value() ? std::to_string(*cand.port()) : "";
      spdlog::info("webrtc ice tx clientId={} sessionId={} mid={} type={} addr={}:{} candBytes={}", clientId, sessionId,
                   mid, static_cast<int>(cand.type()), addr, port, cand_str.size());
    });

    session->pc->onDataChannel([this, clientId, sessionId](std::shared_ptr<rtc::DataChannel> dc) {
      if (!dc)
        return;
      spdlog::info("webrtc datachannel rx clientId={} sessionId={} label={}", clientId, sessionId, dc->label());
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second)
          it->second->dc = dc;
      }
      dc->onOpen([this, clientId, sessionId]() {
        spdlog::info("webrtc dc open clientId={} sessionId={}", clientId, sessionId);
        json out;
        out["type"] = "webrtc.debug";
        out["sessionId"] = sessionId;
        out["event"] = "dcOpen";
        out["ts"] = f8::cppsdk::now_ms();
        enqueue_ws_send(clientId, out.dump());
      });
      dc->onClosed([this, clientId, sessionId]() {
        spdlog::info("webrtc dc closed clientId={} sessionId={}", clientId, sessionId);
        json out;
        out["type"] = "webrtc.debug";
        out["sessionId"] = sessionId;
        out["event"] = "dcClosed";
        out["ts"] = f8::cppsdk::now_ms();
        enqueue_ws_send(clientId, out.dump());
      });
      dc->onMessage([this, clientId, sessionId, dc](rtc::message_variant data) {
        if (const auto* text = std::get_if<std::string>(&data)) {
          spdlog::info("webrtc dc msg rx clientId={} sessionId={} bytes={}", clientId, sessionId, text->size());
          json echo;
          echo["type"] = "webrtc.debug";
          echo["sessionId"] = sessionId;
          echo["event"] = "dcMessage";
          echo["text"] = *text;
          echo["ts"] = f8::cppsdk::now_ms();
          enqueue_ws_send(clientId, echo.dump());
          try {
            dc->send(*text);
          } catch (...) {}
        }
      });
    });

    auto configure_video_track = [this](const std::string& sessionId, const std::shared_ptr<rtc::Track>& t) {
      if (!t)
        return;

      bool has_vp8 = false;
      bool has_h264 = false;
      std::string codec_list;
      std::unordered_map<int, VideoCodec> pt_to_codec;
      std::vector<int> vp8_pts;
      std::vector<int> h264_pts;
      try {
        const auto d = t->description();
        if (d.type() != "video") {
          return;
        }
        for (const int pt : d.payloadTypes()) {
          const auto* m = d.rtpMap(pt);
          if (!m)
            continue;
          std::string fmt = m->format;
          std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                         [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
          if (!codec_list.empty()) {
            codec_list += ",";
          }
          codec_list += fmt;
          if (fmt == "VP8") {
            has_vp8 = true;
            pt_to_codec[pt] = VideoCodec::VP8;
            vp8_pts.push_back(pt);
          }
          if (fmt == "H264") {
            has_h264 = true;
            pt_to_codec[pt] = VideoCodec::H264;
            h264_pts.push_back(pt);
          }
        }
      } catch (...) {
        has_vp8 = false;
        has_h264 = false;
      }

      // Prefer the codec we advertised in the answer (negotiated), not just what the remote offered.
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second) {
          std::lock_guard<std::mutex> lk(it->second->video_mu);
          if (it->second->selected_video_codec != VideoCodec::None && it->second->selected_video_pt >= 0) {
            has_vp8 = it->second->selected_video_codec == VideoCodec::VP8;
            has_h264 = it->second->selected_video_codec == VideoCodec::H264;
            pt_to_codec.clear();
            vp8_pts.clear();
            h264_pts.clear();
            pt_to_codec[it->second->selected_video_pt] = it->second->selected_video_codec;
            if (it->second->selected_video_codec == VideoCodec::VP8) {
              vp8_pts.push_back(it->second->selected_video_pt);
            } else if (it->second->selected_video_codec == VideoCodec::H264) {
              h264_pts.push_back(it->second->selected_video_pt);
            }
          }
        }
      }

      if (!has_vp8 && !has_h264) {
        spdlog::warn("webrtc video track unsupported sessionId={} mid={} codecs={}", sessionId, t->mid(), codec_list);
        video_unsupported_tracks_.fetch_add(1, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> lk(video_err_mu_);
          video_last_error_ = "unsupported_video_codec codecs=" + codec_list;
        }
        return;
      }

      try {
        auto root = std::make_shared<rtc::RtcpReceivingSession>();
        root->addToChain(
            std::make_shared<RtpPacketCounter>(sessionId, &video_rtp_packets_rx_, &video_rtp_bytes_rx_, &video_last_rtp_pt_));
        if (has_vp8) {
          root->addToChain(std::make_shared<Vp8RtpDepacketizer>(vp8_pts));
        } else if (has_h264) {
          auto sink = [this, sessionId](rtc::binary&& data, std::uint8_t pt, std::uint32_t, bool is_keyframe) {
            const std::int64_t now = f8::cppsdk::now_ms();
            const auto prev = video_frames_rx_.fetch_add(1, std::memory_order_relaxed);
            video_last_frame_bytes_.store(static_cast<std::uint64_t>(data.size()), std::memory_order_relaxed);
            video_last_frame_ts_ms_.store(now, std::memory_order_relaxed);
            if (prev == 0) {
              spdlog::info("webrtc video frame rx first sessionId={} bytes={}", sessionId, data.size());
            } else if (((prev + 1) % 120) == 0) {
              spdlog::info("webrtc video frame rx sessionId={} frames={} bytes={}", sessionId, (prev + 1), data.size());
            }

            std::lock_guard<std::mutex> lock(rtc_mu_);
            auto it = sessions_by_id_.find(sessionId);
            if (it == sessions_by_id_.end() || !it->second) {
              return;
            }
            std::lock_guard<std::mutex> lk(it->second->video_mu);
            if (it->second->h264_waiting_for_keyframe && !is_keyframe) {
              return;
            }
            if (is_keyframe) {
              it->second->h264_waiting_for_keyframe = false;
            }
            static constexpr std::size_t kMaxQueue = 30;
            while (it->second->video_queue.size() >= kMaxQueue) {
              it->second->video_queue.pop_front();
            }
            it->second->video_queue.push_back(WebRtcSession::QueuedVideoFrame{std::move(data), VideoCodec::H264,
                                                                              static_cast<int>(pt), now, is_keyframe});
          };
          root->addToChain(std::make_shared<H264RtpDepacketizer>(h264_pts, std::move(sink)));
        }
        t->setMediaHandler(std::move(root));
      } catch (...) {
      }

      if (!codec_list.empty()) {
        spdlog::info("webrtc video track configured sessionId={} mid={} codecs={}", sessionId, t->mid(), codec_list);
      }

      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second) {
          std::lock_guard<std::mutex> lk(it->second->video_mu);
          it->second->remote_video = t;
          it->second->video_mid = t->mid();
          it->second->video_pt_to_codec = std::move(pt_to_codec);
          it->second->video_queue.clear();
          it->second->pending = false;
          it->second->pending_is_keyframe = false;
          it->second->vp8_waiting_for_keyframe = has_vp8;
          it->second->h264_waiting_for_keyframe = has_h264;
          it->second->pending_pt = -1;
          it->second->last_keyframe_request_ms = 0;
          it->second->last_wait_keyframe_log_ms = 0;
          if (has_vp8) {
            it->second->vp8_decoder.reset();
          }
        }
      }

      const VideoCodec fallback_codec = has_vp8 ? VideoCodec::VP8 : VideoCodec::H264;
      t->onFrame([this, sessionId, fallback_codec](rtc::binary data, rtc::FrameInfo fi) {
        const std::int64_t now = f8::cppsdk::now_ms();
        const auto prev = video_frames_rx_.fetch_add(1, std::memory_order_relaxed);
        video_last_frame_bytes_.store(static_cast<std::uint64_t>(data.size()), std::memory_order_relaxed);
        video_last_frame_ts_ms_.store(now, std::memory_order_relaxed);
        if (prev == 0) {
          spdlog::info("webrtc video frame rx first sessionId={} bytes={}", sessionId, data.size());
        } else if (((prev + 1) % 120) == 0) {
          spdlog::info("webrtc video frame rx sessionId={} frames={} bytes={}", sessionId, (prev + 1), data.size());
        }
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it == sessions_by_id_.end() || !it->second) {
          return;
        }
        std::lock_guard<std::mutex> lk(it->second->video_mu);
        const int pt = static_cast<int>(fi.payloadType);
        VideoCodec codec = fallback_codec;
        auto p2c = it->second->video_pt_to_codec.find(pt);
        if (p2c != it->second->video_pt_to_codec.end()) {
          codec = p2c->second;
        }
        bool is_keyframe = false;
        if (codec == VideoCodec::VP8) {
          is_keyframe = looks_like_vp8_keyframe(data);
          if (it->second->vp8_waiting_for_keyframe && !is_keyframe) {
            return;
          }
          if (is_keyframe) {
            it->second->vp8_waiting_for_keyframe = false;
          }
        } else if (codec == VideoCodec::H264) {
          rtc::binary annexb;
          if (OpenH264Decoder::looks_like_annexb(data)) {
            annexb = data;
          } else {
            (void)OpenH264Decoder::avcc_to_annexb(data, annexb);
          }
          is_keyframe = looks_like_h264_idr_annexb(annexb);
          if (it->second->h264_waiting_for_keyframe && !is_keyframe) {
            return;
          }
          if (is_keyframe) {
            it->second->h264_waiting_for_keyframe = false;
          }
        }

        static constexpr std::size_t kMaxQueue = 30;
        while (it->second->video_queue.size() >= kMaxQueue) {
          it->second->video_queue.pop_front();
        }
        it->second->video_queue.push_back(
            WebRtcSession::QueuedVideoFrame{std::move(data), codec, pt, now, is_keyframe});
      });

      const std::int64_t kf_now = f8::cppsdk::now_ms();
      try {
        (void)t->requestKeyframe();
      } catch (...) {
      }
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second) {
          std::lock_guard<std::mutex> lk(it->second->video_mu);
          it->second->last_keyframe_request_ms = kf_now;
        }
      }
    };

    session->pc->onTrack([this, clientId, sessionId, configure_video_track](std::shared_ptr<rtc::Track> tr) {
      if (!tr)
        return;
      try {
        const rtc::Description::Media desc = tr->description();
        spdlog::info("webrtc track rx clientId={} sessionId={} mid={} type={} dir={}", clientId, sessionId, tr->mid(),
                     desc.type(), static_cast<int>(desc.direction()));
        tr->onOpen([clientId, sessionId, mid = tr->mid()]() {
          spdlog::info("webrtc track open clientId={} sessionId={} mid={}", clientId, sessionId, mid);
        });
        tr->onClosed([clientId, sessionId, mid = tr->mid()]() {
          spdlog::info("webrtc track closed clientId={} sessionId={} mid={}", clientId, sessionId, mid);
        });
        tr->onError([clientId, sessionId, mid = tr->mid()](const rtc::string& err) {
          spdlog::warn("webrtc track error clientId={} sessionId={} mid={} err={}", clientId, sessionId, mid, err);
        });

        if (desc.type() != "video") {
          return;
        }
        configure_video_track(sessionId, tr);
      } catch (...) {
        spdlog::info("webrtc track rx clientId={} sessionId={} mid={}", clientId, sessionId, tr->mid());
        return;
      }
    });

    try {
      rtc::Description remote_desc(sdp, sdpType);
      if (video_force_h264_) {
        std::string ferr;
        if (force_offer_h264_only(remote_desc, ferr)) {
          spdlog::info("webrtc offer munged: force H264 only sessionId={}", sid);
        } else {
          spdlog::warn("webrtc offer munged failed sessionId={} err={}", sid, ferr);
          {
            std::lock_guard<std::mutex> lk(video_err_mu_);
            video_last_error_ = "offer_munge_failed: " + ferr;
          }
        }
      }

      // Provide local receive capabilities before applying the remote offer so libdatachannel does not reject m-lines
      // in the generated answer (m=... 0). IMPORTANT: keep the returned shared_ptr alive in the session.
      std::optional<int> vp8_pt;
      std::optional<int> h264_pt;
      try {
        const auto audio_mid = find_first_media_mid(remote_desc, "audio").value_or("0");
        const auto opus_pt = find_first_payload_type(remote_desc, "audio", "OPUS");
        if (opus_pt.has_value()) {
          rtc::Description::Audio audio(audio_mid, rtc::Description::Direction::RecvOnly);
          audio.addOpusCodec(*opus_pt);
          session->local_audio = pc->addTrack(audio);
        }
      } catch (...) {
      }
      try {
        const auto video_mid = find_first_media_mid(remote_desc, "video").value_or("1");
        rtc::Description::Video video(video_mid, rtc::Description::Direction::RecvOnly);
        vp8_pt = find_first_payload_type(remote_desc, "video", "VP8");
        h264_pt = find_preferred_h264_payload_type_from_sdp(sdp, remote_desc);
        if (!h264_pt.has_value()) {
          h264_pt = find_first_payload_type(remote_desc, "video", "H264");
        }

        // IMPORTANT: advertise only one video codec in the answer to avoid ambiguous depacketization/decoding
        // when the offer contains multiple formats (Chrome frequently offers VP8+H264+RTX).
        const bool prefer_h264 = video_force_h264_;
        const bool choose_h264 = prefer_h264 ? h264_pt.has_value() : (!vp8_pt.has_value() && h264_pt.has_value());
        if (choose_h264) {
          video.addH264Codec(*h264_pt);
          session->selected_video_codec = VideoCodec::H264;
          session->selected_video_pt = *h264_pt;
          if (auto fmtp = find_fmtp_for_payload_type_from_sdp(sdp, "video", *h264_pt); fmtp.has_value()) {
            h264_cache_param_sets_from_fmtp(*fmtp, session->h264_sps, session->h264_pps);
          }
        } else if (vp8_pt.has_value()) {
          video.addVP8Codec(*vp8_pt);
          session->selected_video_codec = VideoCodec::VP8;
          session->selected_video_pt = *vp8_pt;
        }
        session->local_video = pc->addTrack(video);
      } catch (...) {
      }

      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        sessions_by_id_[sid] = std::move(session);
      }

      // In libdatachannel, a recvonly Track created with addTrack() is itself the receiving endpoint and may not
      // trigger onTrack(). Configure the local receiving track as well.
      {
        std::shared_ptr<rtc::Track> local_video;
        {
          std::lock_guard<std::mutex> lock(rtc_mu_);
          auto it = sessions_by_id_.find(sid);
          if (it != sessions_by_id_.end() && it->second) {
            local_video = it->second->local_video;
          }
        }
        if (local_video) {
          configure_video_track(sid, local_video);
        }
      }

      pc->setRemoteDescription(remote_desc);
      if (pc->signalingState() == rtc::PeerConnection::SignalingState::HaveRemoteOffer) {
        pc->setLocalDescription(rtc::Description::Type::Answer);
      } else {
        spdlog::warn("webrtc setLocalDescription skipped sessionId={} signalingState={}({})", sid,
                     static_cast<int>(pc->signalingState()), signaling_state_name(pc->signalingState()));
      }
      pc->gatherLocalCandidates();
    } catch (const std::exception& e) {
      spdlog::error("webrtc offer handle failed sessionId={} err={}", sid, e.what());
      stop_session_by_id(sid, "offer_failed");
      return;
    } catch (...) {
      spdlog::error("webrtc offer handle failed sessionId={}", sid);
      stop_session_by_id(sid, "offer_failed");
      return;
    }

    json ack;
    ack["type"] = "webrtc.accepted";
    ack["sessionId"] = sid;
    ack["ts"] = f8::cppsdk::now_ms();
    enqueue_ws_send(ev.client_id, ack.dump());
    return;
  }
}

void WebRtcGatewayService::stop_session_by_id(const std::string& session_id, const std::string& reason) {
  std::unique_ptr<WebRtcSession> sess;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = sessions_by_id_.find(session_id);
    if (it == sessions_by_id_.end())
      return;
    sess = std::move(it->second);
    sessions_by_id_.erase(it);
  }

  if (!sess)
    return;

  spdlog::info("webrtc stop sessionId={} reason={}", session_id, reason);
  if (sess->use_gst) {
    stop_session_gst(*sess);
  }
  if (sess->pc) {
    try {
      sess->pc->close();
    } catch (...) {}
  }
}

void WebRtcGatewayService::stop_sessions_by_client(const std::string& client_id, const std::string& reason) {
  if (client_id.empty())
    return;
  std::vector<std::string> to_stop;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    for (const auto& kv : sessions_by_id_) {
      if (kv.second && kv.second->client_id == client_id)
        to_stop.push_back(kv.first);
    }
  }
  for (const auto& sid : to_stop)
    stop_session_by_id(sid, reason);
}

void WebRtcGatewayService::set_active_local(bool active, const nlohmann::json& meta) {
  active_.store(active, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = published_state_.find("active");
    if (it == published_state_.end() || it->second != active) {
      published_state_["active"] = active;
      f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "active", active, "cmd", meta);
    }
  }
}

void WebRtcGatewayService::on_activate(const nlohmann::json& meta) {
  set_active_local(true, meta);
}
void WebRtcGatewayService::on_deactivate(const nlohmann::json& meta) {
  set_active_local(false, meta);
}
void WebRtcGatewayService::on_set_active(bool active, const nlohmann::json& meta) {
  set_active_local(active, meta);
}

bool WebRtcGatewayService::on_set_state(const std::string& node_id, const std::string& field,
                                        const nlohmann::json& value, const nlohmann::json& meta,
                                        std::string& error_code, std::string& error_message) {
  if (node_id != cfg_.service_id) {
    error_code = "INVALID_ARGS";
    error_message = "nodeId must equal serviceId for service node state";
    return false;
  }

  const std::string f = field;
  if (f == "active") {
    if (!value.is_boolean()) {
      error_code = "INVALID_VALUE";
      error_message = "active must be boolean";
      return false;
    }
    set_active_local(value.get<bool>(), meta);
    return true;
  }
  if (f == "wsPort") {
    if (!value.is_number_integer() && !value.is_number()) {
      error_code = "INVALID_VALUE";
      error_message = "wsPort must be a number";
      return false;
    }
    const auto port_i = static_cast<int>(value.get<double>());
    if (port_i <= 0 || port_i > 65535) {
      error_code = "INVALID_VALUE";
      error_message = "wsPort must be in 1..65535";
      return false;
    }
    cfg_.ws_port = static_cast<std::uint16_t>(port_i);
    std::string err;
    if (!restart_ws(err)) {
      error_code = "INTERNAL";
      error_message = "restart ws failed: " + err;
      return false;
    }
    json write_value = static_cast<int>(cfg_.ws_port);
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      auto it = published_state_.find("wsPort");
      if (it == published_state_.end() || it->second != write_value) {
        published_state_["wsPort"] = write_value;
        f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "wsPort", write_value, "endpoint", meta);
      }
      const auto url = ws_url(cfg_.ws_port);
      auto it2 = published_state_.find("wsUrl");
      if (it2 == published_state_.end() || it2->second != url) {
        published_state_["wsUrl"] = url;
        f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "wsUrl", url, "endpoint", meta);
      }
    }
    return true;
  }
  if (f == "videoEnabled") {
    if (!value.is_boolean()) {
      error_code = "INVALID_VALUE";
      error_message = "videoEnabled must be boolean";
      return false;
    }
    video_enabled_ = value.get<bool>();
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      published_state_["videoEnabled"] = video_enabled_;
    }
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoEnabled", video_enabled_, "endpoint", meta);
    return true;
  }
  if (f == "videoForceH264") {
    if (!value.is_boolean()) {
      error_code = "INVALID_VALUE";
      error_message = "videoForceH264 must be boolean";
      return false;
    }
    video_force_h264_ = value.get<bool>();
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      published_state_["videoForceH264"] = video_force_h264_;
    }
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoForceH264", video_force_h264_, "endpoint",
                                  meta);
    return true;
  }
  if (f == "videoShmMaxWidth" || f == "videoShmMaxHeight" || f == "videoShmMaxFps") {
    if (!value.is_number_integer() && !value.is_number()) {
      error_code = "INVALID_VALUE";
      error_message = f + " must be a number";
      return false;
    }
    const int v = static_cast<int>(value.get<double>());
    if (v < 0) {
      error_code = "INVALID_VALUE";
      error_message = f + " must be >= 0";
      return false;
    }
    if (f == "videoShmMaxWidth")
      video_shm_max_width_ = v;
    else if (f == "videoShmMaxHeight")
      video_shm_max_height_ = v;
    else
      video_shm_max_fps_ = v;

    {
      std::lock_guard<std::mutex> lock(state_mu_);
      published_state_[f] = v;
    }
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, f, v, "endpoint", meta);
    return true;
  }

  error_code = "UNKNOWN_FIELD";
  error_message = "unknown state field";
  return false;
}

bool WebRtcGatewayService::on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta,
                                           std::string& error_code, std::string& error_message) {
  error_code.clear();
  error_message.clear();

  try {
    if (!graph_obj.is_object() || !graph_obj.contains("nodes") || !graph_obj["nodes"].is_array()) {
      return true;
    }

    json service_node;
    for (const auto& n : graph_obj["nodes"]) {
      if (!n.is_object())
        continue;
      const std::string nid = n.value("nodeId", "");
      if (nid != cfg_.service_id)
        continue;
      bool is_service_snapshot = true;
      if (n.contains("operatorClass") && !n["operatorClass"].is_null()) {
        try {
          const std::string oc = n["operatorClass"].is_string() ? n["operatorClass"].get<std::string>() : "";
          if (!oc.empty())
            is_service_snapshot = false;
        } catch (...) {}
      }
      if (!is_service_snapshot)
        continue;
      service_node = n;
      break;
    }

    if (!service_node.is_object() || !service_node.contains("stateValues") ||
        !service_node["stateValues"].is_object()) {
      return true;
    }

    json meta2 = meta;
    if (!meta2.is_object())
      meta2 = json::object();
    meta2["via"] = "rungraph";
    meta2["graphId"] = graph_obj.value("graphId", "");

    const auto& values = service_node["stateValues"];
    for (auto it = values.begin(); it != values.end(); ++it) {
      const std::string field = it.key();
      if (field != "active" && field != "wsPort" && field != "videoEnabled" && field != "videoShmMaxWidth" &&
          field != "videoShmMaxHeight" && field != "videoShmMaxFps" && field != "videoForceH264")
        continue;
      std::string ec;
      std::string em;
      (void)on_set_state(cfg_.service_id, field, it.value(), meta2, ec, em);
    }

    publish_static_state();
    publish_dynamic_state();
  } catch (...) {}

  return true;
}

bool WebRtcGatewayService::on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                                      nlohmann::json& result, std::string& error_code, std::string& error_message) {
  (void)meta;
  error_code.clear();
  error_message.clear();

  if (call == "broadcast") {
    if (!args.is_object() || !args.contains("text") || !args["text"].is_string()) {
      error_code = "INVALID_ARGS";
      error_message = "missing text";
      return false;
    }
    ws_.broadcastText(args["text"].get<std::string>());
    result = json{{"ok", true}};
    return true;
  }
  if (call == "send") {
    if (!args.is_object() || !args.contains("clientId") || !args["clientId"].is_string() || !args.contains("text") ||
        !args["text"].is_string()) {
      error_code = "INVALID_ARGS";
      error_message = "missing clientId/text";
      return false;
    }
    const bool ok = ws_.sendText(args["clientId"].get<std::string>(), args["text"].get<std::string>());
    result = json{{"ok", ok}};
    return ok;
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void WebRtcGatewayService::publish_static_state() {
  const json meta = json{{"via", "startup"}};

  std::vector<std::pair<std::string, json>> updates;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto want = [&](const std::string& field, const json& v) {
      auto it = published_state_.find(field);
      if (it != published_state_.end() && it->second == v)
        return;
      published_state_[field] = v;
      updates.emplace_back(field, v);
    };
    want("serviceClass", cfg_.service_class);
    want("active", active_.load());
    want("wsPort", static_cast<int>(cfg_.ws_port));
    want("wsUrl", ws_url(cfg_.ws_port));

    want("videoEnabled", video_enabled_);
    want("videoForceH264", video_force_h264_);
    want("videoShmMaxWidth", video_shm_max_width_);
    want("videoShmMaxHeight", video_shm_max_height_);
    want("videoShmMaxFps", video_shm_max_fps_);

    want("videoShmName", video_shm_.regionName());
    want("videoShmEvent", video_shm_.frameEventName());
  }

  for (const auto& [field, v] : updates) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
  }
}

void WebRtcGatewayService::publish_dynamic_state() {
  const json meta = json{{"via", "periodic"}};
  const auto cnt = static_cast<int>(ws_.connectionCount());
  std::vector<std::pair<std::string, json>> updates;
  std::string best_sid;
  rtc::PeerConnection* best_pc = nullptr;
  std::int64_t best_last_msg = 0;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    for (auto& kv : sessions_by_id_) {
      if (!kv.second || !kv.second->pc)
        continue;
      if (kv.second->last_msg_ms >= best_last_msg) {
        best_last_msg = kv.second->last_msg_ms;
        best_sid = kv.first;
        best_pc = kv.second->pc.get();
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto want = [&](const std::string& field, const json& v) {
      auto it = published_state_.find(field);
      if (it != published_state_.end() && it->second == v)
        return;
      published_state_[field] = v;
      updates.emplace_back(field, v);
    };
    want("connections", cnt);
    want("videoRtpPacketsRx", static_cast<std::int64_t>(video_rtp_packets_rx_.load(std::memory_order_relaxed)));
    want("videoRtpBytesRx", static_cast<std::int64_t>(video_rtp_bytes_rx_.load(std::memory_order_relaxed)));
    want("videoLastRtpPt", static_cast<std::int64_t>(video_last_rtp_pt_.load(std::memory_order_relaxed)));
    want("videoFramesRx", static_cast<std::int64_t>(video_frames_rx_.load(std::memory_order_relaxed)));
    want("videoFramesDecoded", static_cast<std::int64_t>(video_frames_decoded_.load(std::memory_order_relaxed)));
    want("videoFramesWritten", static_cast<std::int64_t>(video_frames_written_.load(std::memory_order_relaxed)));
    want("videoDecodeErrors", static_cast<std::int64_t>(video_decode_errors_.load(std::memory_order_relaxed)));
    want("videoUnsupportedTracks", static_cast<std::int64_t>(video_unsupported_tracks_.load(std::memory_order_relaxed)));
    want("videoLastFrameBytes", static_cast<std::int64_t>(video_last_frame_bytes_.load(std::memory_order_relaxed)));
    want("videoLastFrameTs", static_cast<std::int64_t>(video_last_frame_ts_ms_.load(std::memory_order_relaxed)));
    want("videoLastDecodeMs", static_cast<std::int64_t>(video_last_decode_ms_.load(std::memory_order_relaxed)));
    want("iceTx", static_cast<std::int64_t>(ice_tx_.load(std::memory_order_relaxed)));
    want("iceRx", static_cast<std::int64_t>(ice_rx_.load(std::memory_order_relaxed)));
    if (best_pc) {
      try {
        want("pcSessionId", best_sid);
        want("pcState", static_cast<int>(best_pc->state()));
        want("iceState", static_cast<int>(best_pc->iceState()));
        want("signalingState", static_cast<int>(best_pc->signalingState()));
        want("pcBytesSent", static_cast<std::int64_t>(best_pc->bytesSent()));
        want("pcBytesReceived", static_cast<std::int64_t>(best_pc->bytesReceived()));
        want("pcRttMs", best_pc->rtt().has_value() ? static_cast<std::int64_t>(best_pc->rtt()->count()) : 0);
        want("pcLocalAddress", best_pc->localAddress().value_or(""));
        want("pcRemoteAddress", best_pc->remoteAddress().value_or(""));
        rtc::Candidate local;
        rtc::Candidate remote;
        if (best_pc->getSelectedCandidatePair(&local, &remote)) {
          want("pcSelectedLocal", local.candidate());
          want("pcSelectedRemote", remote.candidate());
        }
      } catch (...) {
      }
    }
    {
      std::lock_guard<std::mutex> lk(video_err_mu_);
      if (!video_last_error_.empty()) {
        want("videoLastError", video_last_error_);
      }
    }
    {
      std::lock_guard<std::mutex> lk(ice_err_mu_);
      if (!ice_last_error_.empty()) {
        want("iceLastError", ice_last_error_);
      }
    }
  }
  for (const auto& [field, v] : updates) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
  }
}

json WebRtcGatewayService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.webrtc.gateway";
  service["label"] = "WebRTC Gateway";
  service["version"] = "0.0.1";
  service["description"] = "Localhost WebRTC signaling gateway (WS) for browser capture streams.";
  service["stateFields"] = json::array({
      state_field("active", schema_boolean(), "rw", "Active", "Accept and forward signaling when true.", true),
      state_field("wsPort", schema_integer(), "rw", "WS Port", "Localhost websocket port.", true),
      state_field("wsUrl", schema_string(), "ro", "WS URL", "Computed ws://127.0.0.1:<port>/ws"),
      state_field("connections", schema_integer(), "ro", "Connections", "Current websocket client count.", true),
      state_field("videoRtpPacketsRx", schema_integer(), "ro", "Video RTP Packets RX",
                  "Number of received RTP packets for the selected video track (pre-depacketizer)."),
      state_field("videoRtpBytesRx", schema_integer(), "ro", "Video RTP Bytes RX",
                  "Total received RTP bytes for the selected video track (pre-depacketizer)."),
      state_field("videoLastRtpPt", schema_integer(), "ro", "Video Last RTP PT",
                  "Last received RTP payloadType for the video track."),
      state_field("videoEnabled", schema_boolean(), "rw", "Video Enabled", "Decode inbound video to VideoSHM.", true),
      state_field("videoForceH264", schema_boolean(), "rw", "Force H264", "Munge the offer to negotiate H264 only.", true),
      state_field("videoShmMaxWidth", schema_integer(), "rw", "Video Max Width", "Max width for SHM downsample (0 = source)."),
      state_field("videoShmMaxHeight", schema_integer(), "rw", "Video Max Height", "Max height for SHM downsample (0 = source)."),
      state_field("videoShmMaxFps", schema_integer(), "rw", "Video Max FPS", "Max write rate to SHM (0 = unlimited)."),
      state_field("videoShmName", schema_string(), "ro", "Video SHM Name", "Shared memory mapping name for BGRA frames.", true),
      state_field("videoShmEvent", schema_string(), "ro", "Video SHM Event", "Named event signaled on new frame (Windows)."),
      state_field("videoCodec", schema_string(), "ro", "Video Codec", "Decoded video codec (e.g. H264, VP8)."),
      state_field("videoSrcWidth", schema_integer(), "ro", "Video Src Width", "Source frame width (decoded)."),
      state_field("videoSrcHeight", schema_integer(), "ro", "Video Src Height", "Source frame height (decoded)."),
      state_field("videoWidth", schema_integer(), "ro", "Video Width", "SHM frame width (BGRA)."),
      state_field("videoHeight", schema_integer(), "ro", "Video Height", "SHM frame height (BGRA)."),
      state_field("videoPitch", schema_integer(), "ro", "Video Pitch", "SHM frame pitch/stride in bytes."),
      state_field("videoSessionId", schema_string(), "ro", "Video Session", "Session id currently feeding SHM."),
      state_field("videoMid", schema_string(), "ro", "Video MID", "Track mid currently feeding SHM."),
      state_field("videoFramesRx", schema_integer(), "ro", "Video Frames RX", "Number of received video frames (post-depacketizer)."),
      state_field("videoFramesDecoded", schema_integer(), "ro", "Video Frames Decoded", "Number of decoded frames."),
      state_field("videoFramesWritten", schema_integer(), "ro", "Video Frames Written", "Number of frames written to VideoSHM."),
      state_field("videoDecodeErrors", schema_integer(), "ro", "Video Decode Errors", "Decode error count."),
      state_field("videoUnsupportedTracks", schema_integer(), "ro", "Video Unsupported", "Unsupported video track count."),
      state_field("videoLastFrameBytes", schema_integer(), "ro", "Video Last Frame Bytes", "Last received frame byte size."),
      state_field("videoLastFrameTs", schema_integer(), "ro", "Video Last Frame TS", "Last received frame timestamp (ms)."),
      state_field("videoLastDecodeMs", schema_integer(), "ro", "Video Last Decode (ms)", "Last decode time (ms)."),
      state_field("videoLastError", schema_string(), "ro", "Video Last Error", "Last video pipeline error (best-effort)."),
      state_field("iceTx", schema_integer(), "ro", "ICE TX", "Number of local ICE candidates sent to browser."),
      state_field("iceRx", schema_integer(), "ro", "ICE RX", "Number of remote ICE candidates received from browser."),
      state_field("iceLastError", schema_string(), "ro", "ICE Last Error", "Last ICE candidate error (best-effort)."),
      state_field("pcSessionId", schema_string(), "ro", "PC Session", "Most recent session id (periodic snapshot)."),
      state_field("pcState", schema_integer(), "ro", "PC State", "PeerConnection state (numeric)."),
      state_field("iceState", schema_integer(), "ro", "ICE State", "ICE state (numeric)."),
      state_field("signalingState", schema_integer(), "ro", "Signaling", "Signaling state (numeric)."),
      state_field("pcBytesSent", schema_integer(), "ro", "PC Bytes Sent", "Total bytes sent by PeerConnection."),
      state_field("pcBytesReceived", schema_integer(), "ro", "PC Bytes RX", "Total bytes received by PeerConnection."),
      state_field("pcRttMs", schema_integer(), "ro", "PC RTT (ms)", "Best-effort round trip time."),
      state_field("pcLocalAddress", schema_string(), "ro", "PC Local Addr", "Local selected address (best-effort)."),
      state_field("pcRemoteAddress", schema_string(), "ro", "PC Remote Addr", "Remote selected address (best-effort)."),
      state_field("pcSelectedLocal", schema_string(), "ro", "PC Local Cand", "Selected local ICE candidate (best-effort)."),
      state_field("pcSelectedRemote", schema_string(), "ro", "PC Remote Cand", "Selected remote ICE candidate (best-effort)."),
  });

  service["dataOutPorts"] = json::array({
      json{{"name", "signalRx"},
           {"valueSchema", schema_object(json{{"clientId", schema_string()},
                                              {"text", schema_string()},
                                              {"wsUrl", schema_string()},
                                              {"ts", schema_integer()},
                                              {"json", json{{"type", "any"}}}},
                                         json::array({"clientId", "text", "ts"}))},
           {"description", "Inbound websocket signaling messages from browser."},
           {"required", false}},
  });

  service["commands"] = json::array({
      json{{"name", "broadcast"},
           {"description", "Broadcast a text frame to all websocket clients."},
           {"params", json::array({json{{"name", "text"}, {"valueSchema", schema_string()}, {"required", true}}})}},
      json{{"name", "send"},
           {"description", "Send a text frame to one websocket client."},
           {"params", json::array({json{{"name", "clientId"}, {"valueSchema", schema_string()}, {"required", true}},
                                   json{{"name", "text"}, {"valueSchema", schema_string()}, {"required", true}}})}},
  });

  json out;
  out["service"] = service;
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::webrtc_gateway
