#include "implayer_service.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/zenoh_naming.h"
#include "implayer_gui.h"
#include "mpv_player.h"
#include "openxr_presenter.h"
#include "sdl_video_window.h"
#include "video_frame_sink.h"

#if defined(_WIN32)
#define F8_POPEN _popen
#define F8_PCLOSE _pclose
#else
#define F8_POPEN popen
#define F8_PCLOSE pclose
#endif

namespace f8::implayer {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_boolean;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_object;
using f8::cppsdk::describe::schema_string;
using f8::cppsdk::describe::schema_string_enum;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::video_frame_port;

namespace {

bool parse_finite_number(const json& value, const char* name, double& parsed, std::string& err) {
  if (value.is_number()) {
    parsed = value.get<double>();
  } else if (value.is_string()) {
    const std::string text = value.get<std::string>();
    std::size_t consumed = 0;
    try {
      parsed = std::stod(text, &consumed);
    } catch (const std::invalid_argument&) {
      err = std::string("invalid ") + name;
      return false;
    } catch (const std::out_of_range&) {
      err = std::string(name) + " out of range";
      return false;
    }
    if (consumed != text.size()) {
      err = std::string("invalid ") + name;
      return false;
    }
  } else {
    err = std::string("invalid ") + name;
    return false;
  }
  if (!std::isfinite(parsed)) {
    err = std::string(name) + " must be finite";
    return false;
  }
  return true;
}

class ZenohVideoFrameSink final : public VideoFrameSink {
 public:
  explicit ZenohVideoFrameSink(std::shared_ptr<f8::cppsdk::ZenohLatestVideoFramePublisher> publisher)
      : publisher_(std::move(publisher)) {}

  bool ensureConfiguration(unsigned width, unsigned height) override {
    if (width == 0 || height == 0) {
      return false;
    }
    width_ = width;
    height_ = height;
    pitch_ = width * 4u;
    return true;
  }

  bool writeFrame(const void* data, unsigned stride_bytes) override {
    if (!publisher_ || !publisher_->valid() || !data || width_ == 0 || height_ == 0 || pitch_ == 0) {
      return false;
    }
    if (stride_bytes < pitch_) {
      return false;
    }

    const std::byte* payload = static_cast<const std::byte*>(data);
    std::size_t payload_bytes = static_cast<std::size_t>(pitch_) * static_cast<std::size_t>(height_);
    if (stride_bytes != pitch_) {
      scratch_.assign(payload_bytes, std::byte{0});
      for (unsigned y = 0; y < height_; ++y) {
        std::memcpy(scratch_.data() + static_cast<std::size_t>(y) * pitch_,
                    payload + static_cast<std::size_t>(y) * stride_bytes, pitch_);
      }
      payload = scratch_.data();
      payload_bytes = scratch_.size();
    }

    f8::cppsdk::VideoFrameView frame;
    frame.width = width_;
    frame.height = height_;
    frame.pitch = pitch_;
    frame.format = f8::cppsdk::kVideoFormatBgra32;
    frame.frame_id = frame_id_ + 1;
    frame.ts_ms = f8::cppsdk::now_ms();
    frame.payload = payload;
    frame.payload_bytes = payload_bytes;
    if (!publisher_->publish_frame(frame)) {
      return false;
    }
    frame_id_ = frame.frame_id;
    return true;
  }

  unsigned outputWidth() const override { return width_; }
  unsigned outputHeight() const override { return height_; }
  unsigned outputPitch() const override { return pitch_; }
  std::uint64_t frameId() const override { return frame_id_; }

 private:
  std::shared_ptr<f8::cppsdk::ZenohLatestVideoFramePublisher> publisher_;
  unsigned width_ = 0;
  unsigned height_ = 0;
  unsigned pitch_ = 0;
  std::uint64_t frame_id_ = 0;
  std::vector<std::byte> scratch_;
};

std::string new_video_id() {
  static std::atomic<std::uint64_t> g_seq{0};
  const auto seq = g_seq.fetch_add(1, std::memory_order_relaxed);
  return std::to_string(static_cast<long long>(f8::cppsdk::now_ms())) + "-" +
         std::to_string(static_cast<unsigned long long>(seq));
}

std::string trim_copy(std::string s) {
  auto is_ws = [](unsigned char ch) {
    return std::isspace(ch) != 0;
  };
  while (!s.empty() && is_ws(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  while (!s.empty() && is_ws(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

bool looks_like_url_without_scheme(const std::string& s) {
  if (s.empty())
    return false;
  if (s.find("://") != std::string::npos)
    return false;

  const char first = s.front();
  if (first == '/' || first == '\\' || first == '.' || first == '~')
    return false;
  if (s.find('\\') != std::string::npos)
    return false;

  // Keep local Windows paths as local files (e.g. C:/xx or D:\xx).
  if (s.size() >= 2 && std::isalpha(static_cast<unsigned char>(s[0])) != 0 && s[1] == ':')
    return false;

  const std::size_t host_end = s.find_first_of("/?#");
  const std::string host = s.substr(0, host_end);
  if (host.empty())
    return false;

  bool has_dot = false;
  bool has_alpha = false;
  for (char ch : host) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalpha(uch) != 0) {
      has_alpha = true;
      continue;
    }
    if (std::isdigit(uch) != 0 || ch == '-' || ch == ':' || ch == '[' || ch == ']')
      continue;
    if (ch == '.') {
      has_dot = true;
      continue;
    }
    return false;
  }
  return has_dot && has_alpha;
}

std::string normalize_url(std::string s) {
  s = trim_copy(std::move(s));
  while (s.size() >= 2) {
    const char a = s.front();
    const char b = s.back();
    const bool match_double = (a == '"') && (b == '"');
    const bool match_single = (a == '\'') && (b == '\'');
    if (!match_double && !match_single)
      break;
    s = trim_copy(s.substr(1, s.size() - 2));
  }
  if (looks_like_url_without_scheme(s)) {
    s = "https://" + s;
  }
  return s;
}

std::vector<std::string> split_drop_payload(const std::string& raw) {
  std::vector<std::string> out;
  std::string cur;
  cur.reserve(raw.size());
  for (char ch : raw) {
    if (ch == '\r')
      continue;
    if (ch == '\n') {
      auto t = trim_copy(cur);
      if (!t.empty())
        out.emplace_back(std::move(t));
      cur.clear();
      continue;
    }
    cur.push_back(ch);
  }
  auto t = trim_copy(cur);
  if (!t.empty())
    out.emplace_back(std::move(t));
  return out;
}

std::string lowercase_ascii(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

bool starts_with_http_scheme(const std::string& s) {
  const std::string lower = lowercase_ascii(s);
  return lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0;
}

std::string url_path_without_query_or_fragment(const std::string& s) {
  const std::size_t query_pos = s.find_first_of("?#");
  if (query_pos == std::string::npos) {
    return s;
  }
  return s.substr(0, query_pos);
}

bool has_any_suffix(const std::string& s, const std::vector<std::string>& suffixes) {
  for (const std::string& suffix : suffixes) {
    if (s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return true;
    }
  }
  return false;
}

bool looks_like_direct_media_url(const std::string& s) {
  const std::string path = lowercase_ascii(url_path_without_query_or_fragment(s));
  static const std::vector<std::string> kMediaSuffixes = {
      ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".m3u8", ".mpd", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
      ".flac", ".wav"};
  return has_any_suffix(path, kMediaSuffixes);
}

bool should_resolve_with_ytdlp(const std::string& s) {
  if (!starts_with_http_scheme(s)) {
    return false;
  }
  return !looks_like_direct_media_url(s);
}

std::string shell_quote(const std::string& value) {
#if defined(_WIN32)
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') {
      out += "\"\"";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('"');
  return out;
#else
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
#endif
}

std::string ytdlp_executable() {
  std::vector<std::filesystem::path> candidates;
  const char* env_path = std::getenv("F8_IMPLAYER_YTDLP");
  if (env_path != nullptr && env_path[0] != '\0') {
    candidates.emplace_back(env_path);
  }

  const char* base_path_raw = SDL_GetBasePath();
  if (base_path_raw != nullptr) {
    const std::filesystem::path base_path(base_path_raw);
    std::filesystem::path current = base_path;
    while (!current.empty()) {
#if defined(_WIN32)
      candidates.push_back(current / ".pixi" / "envs" / "cpp" / "Scripts" / "yt-dlp.exe");
      candidates.push_back(current / ".pixi" / "envs" / "cpp" / "bin" / "yt-dlp.exe");
#else
      candidates.push_back(current / ".pixi" / "envs" / "cpp" / "bin" / "yt-dlp");
#endif
      const std::filesystem::path parent = current.parent_path();
      if (parent == current) {
        break;
      }
      current = parent;
    }
#if defined(_WIN32)
    candidates.push_back(base_path / "yt-dlp.exe");
#else
    candidates.push_back(base_path / "yt-dlp");
#endif
  }
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
  }
#if defined(_WIN32)
  return "yt-dlp.exe";
#else
  return "yt-dlp";
#endif
}

bool run_command_output(const std::string& command, std::string& output) {
  FILE* pipe = F8_POPEN(command.c_str(), "r");
  if (pipe == nullptr) {
    return false;
  }

  std::array<char, 4096> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = F8_PCLOSE(pipe);
  if (rc != 0) {
    return false;
  }
  return true;
}

struct ResolvedMediaSource {
  std::string video_url;
  std::string audio_url;
  std::vector<std::string> http_headers;
  std::string format_id;
  std::string protocol;
  std::string extractor;
};

struct YtdlpAuthOptions {
  std::string mode = "none";
  std::string browser;
  std::string browser_profile;
  std::string cookies_file;
};

std::string ytdlp_browser_cookie_value(const YtdlpAuthOptions& auth) {
  if (auth.browser_profile.empty()) {
    return auth.browser;
  }
  return auth.browser + ":" + auth.browser_profile;
}

void append_ytdlp_headers(const json& headers, std::vector<std::string>& out) {
  if (!headers.is_object()) {
    return;
  }
  for (auto it = headers.begin(); it != headers.end(); ++it) {
    if (it.key().empty() || !it.value().is_string()) {
      continue;
    }
    const std::string value = it.value().get<std::string>();
    if (!value.empty()) {
      out.push_back(it.key() + ": " + value);
    }
  }
}

bool contains_header_name(const std::vector<std::string>& headers, const std::string& wanted_name) {
  const std::string wanted = lowercase_ascii(wanted_name);
  for (const std::string& header : headers) {
    const std::size_t colon = header.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    if (lowercase_ascii(header.substr(0, colon)) == wanted) {
      return true;
    }
  }
  return false;
}

bool load_resolved_from_json(const json& payload, ResolvedMediaSource& resolved, std::string& err) {
  if (payload.contains("format_id") && payload["format_id"].is_string()) {
    resolved.format_id = payload["format_id"].get<std::string>();
  }
  if (payload.contains("protocol") && payload["protocol"].is_string()) {
    resolved.protocol = payload["protocol"].get<std::string>();
  }
  if (payload.contains("extractor") && payload["extractor"].is_string()) {
    resolved.extractor = payload["extractor"].get<std::string>();
  }

  const json* first_format = nullptr;
  const json* second_format = nullptr;
  if (payload.contains("requested_formats") && payload["requested_formats"].is_array()) {
    const json& requested_formats = payload["requested_formats"];
    if (!requested_formats.empty() && requested_formats[0].is_object()) {
      first_format = &requested_formats[0];
    }
    if (requested_formats.size() >= 2 && requested_formats[1].is_object()) {
      second_format = &requested_formats[1];
    }
  }

  if (first_format != nullptr && first_format->contains("url") && (*first_format)["url"].is_string()) {
    resolved.video_url = (*first_format)["url"].get<std::string>();
    if (first_format->contains("http_headers")) {
      append_ytdlp_headers((*first_format)["http_headers"], resolved.http_headers);
    }
  } else if (payload.contains("url") && payload["url"].is_string()) {
    resolved.video_url = payload["url"].get<std::string>();
    if (payload.contains("http_headers")) {
      append_ytdlp_headers(payload["http_headers"], resolved.http_headers);
    }
  }

  if (second_format != nullptr && second_format->contains("url") && (*second_format)["url"].is_string()) {
    resolved.audio_url = (*second_format)["url"].get<std::string>();
    if (second_format->contains("http_headers") && resolved.http_headers.empty()) {
      append_ytdlp_headers((*second_format)["http_headers"], resolved.http_headers);
    }
  }
  if (payload.contains("webpage_url") && payload["webpage_url"].is_string() &&
      !contains_header_name(resolved.http_headers, "referer")) {
    resolved.http_headers.push_back("Referer: " + payload["webpage_url"].get<std::string>());
  }

  if (resolved.video_url.empty()) {
    err = "yt-dlp resolved no playable media URL";
    return false;
  }
  return true;
}

bool resolve_ytdlp_media(const std::string& url, const YtdlpAuthOptions& auth, ResolvedMediaSource& resolved,
                         std::string& err) {
  const std::string exe = ytdlp_executable();
  const std::string format = "bestvideo[height<=2160]+bestaudio/bestvideo+bestaudio/best";
  const std::string stderr_sink =
#if defined(_WIN32)
      " 2>NUL";
#else
      " 2>/dev/null";
#endif
  std::string command = shell_quote(exe) + " --no-playlist --no-warnings -f " + shell_quote(format) + " -J ";
  if (auth.mode == "browser") {
    command += "--cookies-from-browser " + shell_quote(ytdlp_browser_cookie_value(auth)) + " ";
  } else if (auth.mode == "cookiesFile") {
    command += "--cookies " + shell_quote(auth.cookies_file) + " ";
  }
  command += shell_quote(url) + stderr_sink;
  std::string output;
  if (!run_command_output(command, output) || output.empty()) {
    err = "yt-dlp failed to resolve media URL";
    return false;
  }
  try {
    const json payload = json::parse(output);
    if (!load_resolved_from_json(payload, resolved, err)) {
      return false;
    }
    spdlog::info("yt-dlp resolved media extractor={} format={} protocol={} audio={} headers={} exe={}",
                 resolved.extractor.empty() ? "unknown" : resolved.extractor,
                 resolved.format_id.empty() ? "unknown" : resolved.format_id,
                 resolved.protocol.empty() ? "unknown" : resolved.protocol, resolved.audio_url.empty() ? "no" : "yes",
                 resolved.http_headers.size(), exe);
    return true;
  } catch (const json::exception& exc) {
    err = std::string("yt-dlp returned invalid JSON: ") + exc.what();
    return false;
  }
}

bool open_player_media(MpvPlayer& player, const std::string& url, const YtdlpAuthOptions& auth, std::string& err) {
  if (!should_resolve_with_ytdlp(url)) {
    if (!player.openMedia(url, "", {})) {
      err = "mpv loadfile failed";
      return false;
    }
    return true;
  }

  ResolvedMediaSource resolved;
  if (!resolve_ytdlp_media(url, auth, resolved, err)) {
    return false;
  }
  if (!player.openMedia(resolved.video_url, resolved.audio_url, resolved.http_headers)) {
    err = "mpv loadfile failed";
    return false;
  }
  return true;
}

bool parse_auth_mode(const std::string& raw, std::string& normalized_mode) {
  const std::string mode = lowercase_ascii(trim_copy(raw));
  if (mode == "none") {
    normalized_mode = "none";
    return true;
  }
  if (mode == "browser") {
    normalized_mode = "browser";
    return true;
  }
  if (mode == "cookiesfile") {
    normalized_mode = "cookiesFile";
    return true;
  }
  return false;
}

bool is_supported_auth_mode(const std::string& mode) {
  std::string normalized;
  return parse_auth_mode(mode, normalized);
}

bool parse_openxr_mode(const std::string& raw, std::string& normalized_mode) {
  const std::string mode = lowercase_ascii(trim_copy(raw));
  if (mode == "off" || mode == "0" || mode == "false" || mode == "disabled") {
    normalized_mode = "off";
    return true;
  }
  if (mode == "on" || mode == "1" || mode == "true" || mode == "enabled") {
    normalized_mode = "on";
    return true;
  }
  if (mode == "auto") {
    normalized_mode = "auto";
    return true;
  }
  return false;
}

bool is_supported_auth_browser(const std::string& browser) {
  return browser == "chrome" || browser == "chromium" || browser == "edge" || browser == "firefox" ||
         browser == "safari";
}

bool is_sensitive_auth_field(const std::string& field) {
  return field == "authBrowserProfile" || field == "authCookiesFile";
}

bool is_profile_value_safe(const std::string& profile) {
  // ytdl-raw-options uses comma as option separator; disallow commas/newlines to
  // keep parsing deterministic.
  return profile.find(',') == std::string::npos && profile.find('\n') == std::string::npos &&
         profile.find('\r') == std::string::npos;
}

float normalize_yaw_deg(float yaw_deg) {
  float v = std::fmod(yaw_deg, 360.0f);
  if (v > 180.0f)
    v -= 360.0f;
  if (v < -180.0f)
    v += 360.0f;
  return v;
}

MpvPlayer::FrameExportViewMode frame_export_view_mode_for_vr(SdlVideoWindow::ProjectionMode mode, int sbs_eye) {
  if (mode == SdlVideoWindow::ProjectionMode::EquirectSbs) {
    return sbs_eye == 0 ? MpvPlayer::FrameExportViewMode::SbsLeft : MpvPlayer::FrameExportViewMode::SbsRight;
  }
  return MpvPlayer::FrameExportViewMode::FullFrame;
}

}  // namespace

ImPlayerService::ImPlayerService(Config cfg) : cfg_(std::move(cfg)) {}

ImPlayerService::~ImPlayerService() {
  stop();
}

void ImPlayerService::on_lifecycle(bool active, const nlohmann::json&) {
  set_active_local(active);
}

void ImPlayerService::on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                               std::int64_t ts_ms, const nlohmann::json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id)
    return;
  std::string ec;
  std::string em;
  json result;
  (void)on_set_state(node_id, field, value, meta, ec, em);
}

bool ImPlayerService::start() {
  if (running_.load(std::memory_order_acquire))
    return true;

  try {
    cfg_.service_id = f8::cppsdk::ensure_token(cfg_.service_id, "service_id");
  } catch (const std::exception& e) {
    spdlog::error("invalid --service-id: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("invalid --service-id");
    return false;
  }

  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);

  const std::string key = f8::cppsdk::zenoh_data_key(cfg_.service_id, cfg_.service_id, "video");
  auto publisher = std::make_shared<f8::cppsdk::ZenohLatestVideoFramePublisher>();
  if (!publisher->open(runtime_backend, key)) {
    zenoh_video_key_.clear();
    zenoh_video_publisher_.reset();
    frame_sink_.reset();
    spdlog::error("implayer zenoh video publisher unavailable serviceId={} key={}", cfg_.service_id, key);
    return false;
  }
  zenoh_video_key_ = key;
  zenoh_video_publisher_ = publisher;
  frame_sink_ = std::make_shared<ZenohVideoFrameSink>(publisher);
  spdlog::info("implayer zenoh video publisher enabled serviceId={} key={}", cfg_.service_id, key);

  SdlVideoWindow::Config wcfg;
  wcfg.title = "f8implayer - " + cfg_.service_id;
  wcfg.width = cfg_.window_width;
  wcfg.height = cfg_.window_height;
  wcfg.resizable = cfg_.window_resizable;
  wcfg.vsync = cfg_.window_vsync;
#if defined(_WIN32)
  // Create a modern OpenGL context on Windows so OpenXR can be enabled later
  // via state (Studio), even if it wasn't requested via CLI at startup.
  wcfg.gl_major = 4;
  wcfg.gl_minor = 5;
  wcfg.gl_allow_fallback = true;
  wcfg.gl_fallback_major = 3;
  wcfg.gl_fallback_minor = 3;
#endif
  window_ = std::make_unique<SdlVideoWindow>(wcfg);
  if (!window_->start()) {
    spdlog::error("failed to start SDL video window");
    return false;
  }
  if (!window_->makeCurrent()) {
    spdlog::error("failed to activate SDL GL context");
    return false;
  }

  openxr_mirror_window_.store(cfg_.openxr_mirror_window, std::memory_order_release);
  std::string openxr_mode_startup;
  {
    std::string mode;
    if (!parse_openxr_mode(cfg_.openxr_mode, mode)) {
      spdlog::error("invalid --openxr-mode: {}", cfg_.openxr_mode);
      return false;
    }
    openxr_mode_startup = mode;
    std::lock_guard<std::mutex> lock(state_mu_);
    openxr_mode_ = mode;
  }
  openxr_next_retry_ms_ = f8::cppsdk::now_ms();

  const bool want_openxr_now = openxr_mode_startup == "on" || openxr_mode_startup == "auto";
  if (want_openxr_now) {
    openxr_ = std::make_unique<OpenXrPresenter>();
    std::string xr_err;
    if (!openxr_->start(window_->sdlWindow(), window_->glContext(), xr_err)) {
      openxr_.reset();
      if (openxr_mode_startup == "on") {
        spdlog::error("failed to start OpenXR presenter: {}", xr_err);
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        openxr_last_start_error_ = xr_err;
      }
      spdlog::warn("OpenXR not available at startup (mode=auto): {}", xr_err);
    }
  }

  gui_ = std::make_unique<ImPlayerGui>();
  if (!gui_->start(window_->sdlWindow(), window_->glContext())) {
    spdlog::error("failed to initialize ImGui overlay");
    return false;
  }

  MpvPlayer::VideoConfig vcfg;
  vcfg.offline = false;
  vcfg.videoOutputMaxWidth = cfg_.video_output_max_width;
  vcfg.videoOutputMaxHeight = cfg_.video_output_max_height;
  vcfg.videoOutputMaxFps = cfg_.video_output_max_fps;

  try {
    player_ = std::make_unique<MpvPlayer>(
        vcfg,
        [this](double pos, double dur) {
          position_seconds_.store(pos, std::memory_order_relaxed);
          duration_seconds_.store(dur, std::memory_order_relaxed);
        },
        [this](bool playing) {
          playing_.store(playing, std::memory_order_relaxed);
          gui_state_dirty_.store(true, std::memory_order_release);
        },
        [this]() {
          playing_.store(false, std::memory_order_relaxed);
          media_finished_.store(true, std::memory_order_release);
          eof_reached_.store(true, std::memory_order_release);
          gui_state_dirty_.store(true, std::memory_order_release);
        });
  } catch (const std::exception& e) {
    spdlog::error("mpv init failed: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("mpv init failed: unknown error");
    return false;
  }
  player_->setVideoFrameSink(frame_sink_);
  if (!player_->initializeGl()) {
    spdlog::error("failed to initialize mpv GL render context");
    return false;
  }
  player_->setVolume(volume_);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    std::string auth_err;
    if (!apply_auth_options_locked(auth_err)) {
      last_error_ = auth_err;
      spdlog::warn("failed to apply initial auth options: {}", auth_err);
    }
  }

  // Start the service bus only after the GUI/player are ready, so rungraph/state
  // deployments won't race against initialization.
  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.apply_runtime_backend(runtime_backend);
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "IM Player";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_set_state_node(this);
  bus_->add_rungraph_node(this);
  bus_->add_command_node(this, ImPlayerService::describe());
  if (!bus_->start()) {
    bus_.reset();
    if (zenoh_video_publisher_) {
      zenoh_video_publisher_->close();
    }
    zenoh_video_publisher_.reset();
    zenoh_video_key_.clear();
    frame_sink_.reset();
    return false;
  }

  if (!cfg_.initial_media_url.empty()) {
    std::string err;
    if (!cmd_open(json{{"url", cfg_.initial_media_url}}, err)) {
      spdlog::error("initial --media failed: {}", err);
    }
  }

  publish_static_state();
  publish_dynamic_state();

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("implayer started serviceId={} backend={} videoBackend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend), "zenoh");
  return true;
}

void ImPlayerService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  stop_requested_.store(true, std::memory_order_release);

  try {
    if (bus_)
      bus_->stop();
  } catch (const std::exception& exc) {
    spdlog::warn("implayer service bus stop failed serviceId={}: {}", cfg_.service_id, exc.what());
  } catch (...) {
    spdlog::warn("implayer service bus stop failed serviceId={}: unknown error", cfg_.service_id);
  }
  bus_.reset();

  if (openxr_)
    openxr_->stop();
  openxr_.reset();

  if (window_)
    window_->makeCurrent();
  if (player_)
    player_->shutdownGl();
  if (gui_)
    gui_->stop();
  gui_.reset();
  player_.reset();
  window_.reset();
  if (zenoh_video_publisher_) {
    zenoh_video_publisher_->close();
  }
  zenoh_video_publisher_.reset();
  zenoh_video_key_.clear();
  frame_sink_.reset();
}

void ImPlayerService::tick() {
  if (!running_.load(std::memory_order_acquire))
    return;

  bool did_present = false;
  const std::int64_t tick_now_ms = f8::cppsdk::now_ms();
  if (last_tick_ms_ > 0) {
    const double dt_ms = static_cast<double>(std::max<std::int64_t>(1, tick_now_ms - last_tick_ms_));
    constexpr double alpha = 0.12;
    if (tick_ema_ms_ <= 0.0) {
      tick_ema_ms_ = dt_ms;
    } else {
      tick_ema_ms_ = (1.0 - alpha) * tick_ema_ms_ + alpha * dt_ms;
    }
    tick_ema_fps_ = tick_ema_ms_ > 0.0 ? (1000.0 / tick_ema_ms_) : 0.0;
  }
  last_tick_ms_ = tick_now_ms;

  if (bus_) {
    (void)bus_->drain_main_thread();
  }

  if (bus_ && bus_->terminate_requested()) {
    stop_requested_.store(true, std::memory_order_release);
    return;
  }

  if (media_finished_.exchange(false, std::memory_order_acq_rel)) {
    playlist_next();
  }

  if (window_ && window_->wantsClose()) {
    stop_requested_.store(true, std::memory_order_release);
    return;
  }

  if (player_ && window_) {
    std::unique_lock<std::mutex> render_lock(render_mu_, std::try_to_lock);
    if (render_lock.owns_lock()) {
      std::string openxr_mode_snapshot;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        openxr_mode_snapshot = openxr_mode_;
      }
      (void)window_->makeCurrent();
      const bool want_clear = clear_video_requested_.exchange(false, std::memory_order_acq_rel);
      bool force_present = false;
      if (want_clear && player_) {
        player_->resetVideoOutput();
        force_present = true;
      }
      const unsigned vw = player_->videoWidth();
      const unsigned vh = player_->videoHeight();
      if (vw != 0 && vh != 0 && (vw != view_last_video_w_ || vh != view_last_video_h_)) {
        view_last_video_w_ = vw;
        view_last_video_h_ = vh;
        view_zoom_ = 1.0f;
        view_pan_x_ = 0.0f;
        view_pan_y_ = 0.0f;
        view_panning_ = false;
      }
      player_->setFrameExportViewMode(frame_export_view_mode_for_vr(vr_mode_, vr_sbs_eye_));
      const bool updated = player_->renderVideoFrame();

      // OpenXR is controlled by `openxrMode` state; keep the service running even
      // if the headset/runtime is not available (auto mode).
      if (openxr_mode_snapshot == "off") {
        if (openxr_) {
          openxr_->stop();
          openxr_.reset();
        }
      } else if (!openxr_ && tick_now_ms >= openxr_next_retry_ms_) {
        openxr_ = std::make_unique<OpenXrPresenter>();
        std::string xr_err;
        if (!openxr_->start(window_->sdlWindow(), window_->glContext(), xr_err)) {
          openxr_.reset();
          {
            std::lock_guard<std::mutex> lock(state_mu_);
            if (xr_err != openxr_last_start_error_) {
              openxr_last_start_error_ = xr_err;
              last_error_ = xr_err;
            }
          }
          openxr_next_retry_ms_ = tick_now_ms + 2000;
        } else {
          std::lock_guard<std::mutex> lock(state_mu_);
          openxr_last_start_error_.clear();
        }
      }

      const bool mirror_disabled =
          (openxr_ != nullptr) && !openxr_mirror_window_.load(std::memory_order_acquire);
      if (openxr_) {
        OpenXrPresenter::FrameParams fp;
        fp.src_texture = player_->videoTextureId();
        fp.src_width = player_->videoWidth();
        fp.src_height = player_->videoHeight();
        fp.mode = vr_mode_;
        fp.sbs_eye = vr_sbs_eye_;
        fp.playing = playing_.load(std::memory_order_relaxed);
        fp.position_seconds = position_seconds_.load(std::memory_order_relaxed);
        fp.duration_seconds = duration_seconds_.load(std::memory_order_relaxed);
        fp.yaw_offset_deg = vr_yaw_deg_;
        fp.pitch_offset_deg = vr_pitch_deg_;

        OpenXrPresenter::Events xr_events;
        std::string xr_err;
        if (!openxr_->renderFrame(fp, &xr_events, xr_err)) {
          {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = xr_err.empty() ? "OpenXR presenter failed" : xr_err;
          }
          openxr_->stop();
          openxr_.reset();
          openxr_next_retry_ms_ = tick_now_ms + 1000;
        } else if (!xr_err.empty()) {
          std::lock_guard<std::mutex> lock(state_mu_);
          last_error_ = xr_err;
        }

        if (xr_events.cycle_projection_pressed) {
          if (vr_mode_ == SdlVideoWindow::ProjectionMode::Flat2D) {
            vr_mode_ = SdlVideoWindow::ProjectionMode::EquirectMono;
          } else if (vr_mode_ == SdlVideoWindow::ProjectionMode::EquirectMono) {
            vr_mode_ = SdlVideoWindow::ProjectionMode::EquirectSbs;
          } else {
            vr_mode_ = SdlVideoWindow::ProjectionMode::Flat2D;
          }
        }
        if (xr_events.play_pause_pressed) {
          std::string err;
          if (playing_.load(std::memory_order_acquire)) {
            (void)cmd_pause(err);
          } else {
            (void)cmd_play(err);
          }
        }
        if (xr_events.playlist_next_pressed) {
          playlist_next();
        }
        if (xr_events.playlist_prev_pressed) {
          playlist_prev();
        }
        if (xr_events.seek_absolute_valid && duration_seconds_.load(std::memory_order_relaxed) > 0.0) {
          const double dur = duration_seconds_.load(std::memory_order_relaxed);
          const double frac = std::clamp(xr_events.seek_absolute_fraction01, 0.0, 1.0);
          const double pos = frac * dur;
          std::string err;
          (void)cmd_seek(json{{"position", pos}}, err);
        } else if (std::abs(xr_events.seek_delta_seconds) > 1e-6) {
          const double cur = position_seconds_.load(std::memory_order_relaxed);
          const double dur = duration_seconds_.load(std::memory_order_relaxed);
          double next = cur + xr_events.seek_delta_seconds;
          if (dur > 0.0) {
            next = std::clamp(next, 0.0, dur);
          } else {
            next = std::max(0.0, next);
          }
          std::string err;
          (void)cmd_seek(json{{"position", next}}, err);
        }

        did_present = true;
        if (mirror_disabled) {
          // Avoid a busy-spin when OpenXR is active but there is no mirror present.
          if (gui_)
            gui_->clearRepaintFlag();
          window_->clearRedrawFlag();
        }
      }

      if (!mirror_disabled &&
          (force_present || updated || window_->needsRedraw() || (gui_ && gui_->wantsRepaint()) ||
           gui_state_dirty_.exchange(false, std::memory_order_acq_rel))) {
        ImPlayerGui::Callbacks cb;
        cb.open = [this](const std::string& url) {
          std::string err;
          const bool ok = cmd_open(json{{"url", url}}, err);
          report_gui_result("Open", ok, err);
        };
        cb.play = [this]() {
          std::string err;
          const bool ok = cmd_play(err);
          report_gui_result("Play", ok, err);
        };
        cb.pause = [this]() {
          std::string err;
          const bool ok = cmd_pause(err);
          report_gui_result("Pause", ok, err);
        };
        cb.stop = [this]() {
          std::string err;
          const bool ok = cmd_stop(err);
          report_gui_result("Stop", ok, err);
        };
        cb.seek = [this](double pos) {
          std::string err;
          const bool ok = cmd_seek(json{{"position", pos}}, err);
          report_gui_result("Seek", ok, err);
        };
        cb.set_volume = [this](double vol) {
          std::string err;
          const bool ok = cmd_set_volume(json{{"volume", vol}}, err);
          report_gui_result("Volume", ok, err);
        };
        cb.set_loop = [this](bool loop) {
          std::lock_guard<std::mutex> lock(state_mu_);
          loop_ = loop;
        };
        cb.set_hwdec = [this](const std::string& hwdec) {
          if (!player_)
            return;
          if (!player_->setHwdec(hwdec)) {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = "failed to set hwdec=" + hwdec;
          }
        };
        cb.set_hwdec_extra_frames = [this](int extra_frames) {
          if (!player_)
            return;
          if (!player_->setHwdecExtraFrames(extra_frames)) {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = "failed to set hwdec-extra-frames=" + std::to_string(extra_frames);
          }
        };
        cb.set_fbo_format = [this](const std::string& fbo_format) {
          if (!player_)
            return;
          if (!player_->setFboFormat(fbo_format)) {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = "failed to set fbo-format=" + fbo_format;
          }
        };
        cb.fit_view = [this]() {
          view_zoom_ = 1.0f;
          view_pan_x_ = 0.0f;
          view_pan_y_ = 0.0f;
          view_panning_ = false;
        };
        cb.toggle_fullscreen = [this]() {
          if (window_)
            (void)window_->toggleFullscreen();
        };
        cb.set_vr_mode = [this](SdlVideoWindow::ProjectionMode mode) {
          vr_mode_ = mode;
        };
        cb.set_vr_eye = [this](int eye) {
          vr_sbs_eye_ = (eye == 0) ? 0 : 1;
        };
        cb.set_vr_fov = [this](float fov_deg) {
          vr_fov_deg_ = std::clamp(fov_deg, 50.0f, 120.0f);
        };
        cb.reset_vr_view = [this]() {
          vr_yaw_deg_ = 0.0f;
          vr_pitch_deg_ = 0.0f;
          vr_fov_deg_ = 90.0f;
        };
        cb.playlist_select = [this](int index) {
          playlist_play_index(index);
        };
        cb.playlist_remove = [this](int index) {
          playlist_remove_index(index);
        };
        cb.playlist_clear = [this]() {
          playlist_clear();
        };
        cb.playlist_next = [this]() {
          playlist_next();
        };
        cb.playlist_prev = [this]() {
          playlist_prev();
        };

        std::string err;
        std::vector<std::string> playlist_snapshot;
        int playlist_index_snapshot = -1;
        bool loop_snapshot = false;
        double volume_snapshot = 1.0;
        std::string media_url_snapshot;
        {
          std::lock_guard<std::mutex> lock(state_mu_);
          err = last_error_;
          playlist_snapshot = playlist_;
          playlist_index_snapshot = playlist_index_;
          loop_snapshot = loop_;
          volume_snapshot = volume_;
          media_url_snapshot = media_url_;
        }

        const SdlVideoWindow::ViewTransform view{view_zoom_, view_pan_x_, view_pan_y_};
        const SdlVideoWindow::VrViewState vr_view{
            vr_mode_, vr_yaw_deg_, std::clamp(vr_pitch_deg_, -89.0f, 89.0f), std::clamp(vr_fov_deg_, 50.0f, 120.0f),
            vr_sbs_eye_};
        const bool playing = playing_.load(std::memory_order_relaxed);
        window_->present(
            *player_,
            [this, &cb, &err, &playlist_snapshot, playlist_index_snapshot, playing, loop_snapshot,
             volume_snapshot, &media_url_snapshot]() {
              if (gui_ && player_) {
                gui_->renderOverlay(*player_, cb, err, playlist_snapshot, playlist_index_snapshot, playing,
                                    loop_snapshot, volume_snapshot, media_url_snapshot, tick_ema_fps_, tick_ema_ms_,
                                    vr_mode_, vr_sbs_eye_, vr_yaw_deg_,
                                    vr_pitch_deg_, vr_fov_deg_);
              }
            },
            view, vr_view);
        did_present = true;
      }
    }
  }

  const std::int64_t now = tick_now_ms;
  if (now - last_state_pub_ms_ >= 200) {
    publish_dynamic_state();
    last_state_pub_ms_ = now;
  }

  if (now - last_playback_data_pub_ms_ >= 200) {
    last_playback_data_pub_ms_ = now;
    json evt;
    bool have_video = false;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (!video_id_.empty()) {
        evt["videoId"] = video_id_;
        have_video = true;
      }
    }
    if (!have_video) {
      return;
    }
    evt["position"] = position_seconds_.load(std::memory_order_relaxed);
    evt["duration"] = duration_seconds_.load(std::memory_order_relaxed);
    evt["playing"] = playing_.load(std::memory_order_relaxed);
    if (bus_) {
      (void)bus_->emit_data(cfg_.service_id, "playback", evt, now);
    }
  }

  // Avoid a busy-spin when there is nothing to render (e.g. still images),
  // which otherwise can keep a CPU core hot even with vsync enabled.
  if (!did_present && !(window_ && window_->needsRedraw()) && !(gui_ && gui_->wantsRepaint())) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

void ImPlayerService::processSdlEvent(const SDL_Event& ev) {
  if (!running_.load(std::memory_order_acquire))
    return;

  if (window_)
    window_->processEvent(ev);

  SDL_Event copy = ev;
  if (gui_)
    gui_->processEvent(&copy);

  if (ev.type == SDL_EVENT_DROP_FILE || ev.type == SDL_EVENT_DROP_TEXT) {
    if (ev.drop.data) {
      const auto items = split_drop_payload(std::string(ev.drop.data));
      playlist_add(items, true);
    }
    return;
  }

  if (ev.type == SDL_EVENT_MOUSE_WHEEL) {
    if (gui_ && gui_->wantsCaptureMouse()) {
      return;
    }
    const float zoom_step = 1.05f;
    if (ev.wheel.y > 0) {
      view_zoom_ *= zoom_step;
    } else if (ev.wheel.y < 0) {
      view_zoom_ /= zoom_step;
    }
    view_zoom_ = std::clamp(view_zoom_, 0.1f, 10.0f);
    return;
  }

  if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
    if (ev.button.button == SDL_BUTTON_LEFT && vr_mode_ != SdlVideoWindow::ProjectionMode::Flat2D) {
      if (!(gui_ && gui_->wantsCaptureMouse()) && window_) {
        int win_w = 0, win_h = 0;
        int px_w = 0, px_h = 0;
        SDL_GetWindowSize(window_->sdlWindow(), &win_w, &win_h);
        SDL_GetWindowSizeInPixels(window_->sdlWindow(), &px_w, &px_h);
        const float sx = (win_w > 0 && px_w > 0) ? static_cast<float>(px_w) / static_cast<float>(win_w) : 1.0f;
        const float sy = (win_h > 0 && px_h > 0) ? static_cast<float>(px_h) / static_cast<float>(win_h) : 1.0f;
        vr_dragging_ = true;
        vr_drag_anchor_x_ = static_cast<float>(ev.button.x) * sx;
        vr_drag_anchor_y_ = static_cast<float>(ev.button.y) * sy;
        vr_drag_start_yaw_deg_ = vr_yaw_deg_;
        vr_drag_start_pitch_deg_ = vr_pitch_deg_;
      }
      return;
    }
    if (ev.button.button == SDL_BUTTON_MIDDLE) {
      int win_w = 0, win_h = 0;
      int px_w = 0, px_h = 0;
      SDL_GetWindowSize(window_->sdlWindow(), &win_w, &win_h);
      SDL_GetWindowSizeInPixels(window_->sdlWindow(), &px_w, &px_h);
      const float sx = (win_w > 0 && px_w > 0) ? static_cast<float>(px_w) / static_cast<float>(win_w) : 1.0f;
      const float sy = (win_h > 0 && px_h > 0) ? static_cast<float>(px_h) / static_cast<float>(win_h) : 1.0f;

      view_panning_ = true;
      view_pan_anchor_x_ = static_cast<float>(ev.button.x) * sx;
      view_pan_anchor_y_ = static_cast<float>(ev.button.y) * sy;
      view_pan_start_x_ = view_pan_x_;
      view_pan_start_y_ = view_pan_y_;
    }
    return;
  }

  if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP) {
    if (ev.button.button == SDL_BUTTON_LEFT) {
      vr_dragging_ = false;
      return;
    }
    if (ev.button.button == SDL_BUTTON_MIDDLE) {
      view_panning_ = false;
    }
    return;
  }

  if (ev.type == SDL_EVENT_MOUSE_MOTION) {
    if (vr_dragging_ && window_) {
      int win_w = 0, win_h = 0;
      int px_w = 0, px_h = 0;
      SDL_GetWindowSize(window_->sdlWindow(), &win_w, &win_h);
      SDL_GetWindowSizeInPixels(window_->sdlWindow(), &px_w, &px_h);
      const float sx = (win_w > 0 && px_w > 0) ? static_cast<float>(px_w) / static_cast<float>(win_w) : 1.0f;
      const float sy = (win_h > 0 && px_h > 0) ? static_cast<float>(px_h) / static_cast<float>(win_h) : 1.0f;
      const float mx = static_cast<float>(ev.motion.x) * sx;
      const float my = static_cast<float>(ev.motion.y) * sy;
      constexpr float kYawDegPerPixel = 0.12f;
      constexpr float kPitchDegPerPixel = 0.12f;
      vr_yaw_deg_ = normalize_yaw_deg(vr_drag_start_yaw_deg_ + (mx - vr_drag_anchor_x_) * kYawDegPerPixel);
      vr_pitch_deg_ = std::clamp(vr_drag_start_pitch_deg_ - (my - vr_drag_anchor_y_) * kPitchDegPerPixel, -89.0f, 89.0f);
      return;
    }
    if (view_panning_) {
      int win_w = 0, win_h = 0;
      int px_w = 0, px_h = 0;
      SDL_GetWindowSize(window_->sdlWindow(), &win_w, &win_h);
      SDL_GetWindowSizeInPixels(window_->sdlWindow(), &px_w, &px_h);
      const float sx = (win_w > 0 && px_w > 0) ? static_cast<float>(px_w) / static_cast<float>(win_w) : 1.0f;
      const float sy = (win_h > 0 && px_h > 0) ? static_cast<float>(px_h) / static_cast<float>(win_h) : 1.0f;

      const float mx = static_cast<float>(ev.motion.x) * sx;
      const float my = static_cast<float>(ev.motion.y) * sy;
      view_pan_x_ = view_pan_start_x_ + (mx - view_pan_anchor_x_);
      // SDL y+ goes down, OpenGL framebuffer y+ goes up.
      view_pan_y_ = view_pan_start_y_ - (my - view_pan_anchor_y_);
    }
    return;
  }

  if (ev.type == SDL_EVENT_KEY_DOWN) {
    if (!player_)
      return;
    if (gui_ && gui_->wantsCaptureKeyboard())
      return;
    const SDL_Keycode key = ev.key.key;
    if (key == SDLK_F) {
      if (window_)
        (void)window_->toggleFullscreen();
      return;
    }
    if (key == SDLK_ESCAPE) {
      if (window_ && window_->isFullscreen())
        (void)window_->setFullscreen(false);
      return;
    }
    if (key == SDLK_0) {
      view_zoom_ = 1.0f;
      view_pan_x_ = 0.0f;
      view_pan_y_ = 0.0f;
      view_panning_ = false;
      return;
    }
    if (key == SDLK_SPACE) {
      std::string err;
      if (playing_.load(std::memory_order_relaxed)) {
        report_gui_result("Pause", cmd_pause(err), err);
      } else {
        report_gui_result("Play", cmd_play(err), err);
      }
    } else if (key == SDLK_LEFT) {
      const double p = position_seconds_.load(std::memory_order_relaxed);
      std::string err;
      report_gui_result("Seek", cmd_seek(json{{"position", std::max(0.0, p - 5.0)}}, err), err);
    } else if (key == SDLK_RIGHT) {
      const double p = position_seconds_.load(std::memory_order_relaxed);
      std::string err;
      report_gui_result("Seek", cmd_seek(json{{"position", p + 5.0}}, err), err);
    } else if (key == SDLK_UP) {
      double v;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        v = std::min(1.0, volume_ + 0.05);
      }
      std::string err;
      report_gui_result("Volume", cmd_set_volume(json{{"volume", v}}, err), err);
    } else if (key == SDLK_DOWN) {
      double v;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        v = std::max(0.0, volume_ - 0.05);
      }
      std::string err;
      report_gui_result("Volume", cmd_set_volume(json{{"volume", v}}, err), err);
    }
  }
}

PlaybackIntent ImPlayerService::playback_intent() const {
  return playback_intent_.load(std::memory_order_acquire);
}

void ImPlayerService::set_playback_intent(PlaybackIntent intent) {
  playback_intent_.store(intent, std::memory_order_release);
}

void ImPlayerService::set_active_local(bool active) {
  active_.store(active, std::memory_order_release);
}

bool ImPlayerService::on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                                   const nlohmann::json& meta, std::string& error_code, std::string& error_message) {
  if (node_id != cfg_.service_id) {
    error_code = "INVALID_ARGS";
    error_message = "nodeId must equal serviceId for service node state";
    return false;
  }

  const std::string f = field;
  std::string err;
  bool ok = false;
  if (f == "mediaUrl") {
    if (!value.is_string()) {
      err = "mediaUrl must be a string";
      ok = false;
    } else {
      const std::string u = normalize_url(value.get<std::string>());
      if (u.empty()) {
        err = "missing url";
        ok = false;
      } else {
        bool inserted = false;
        int previous_index = -1;
        {
          std::lock_guard<std::mutex> lock(state_mu_);
          previous_index = playlist_index_;
          int existing_index = -1;
          for (std::size_t i = 0; i < playlist_.size(); ++i) {
            if (playlist_[i] == u) {
              existing_index = static_cast<int>(i);
              break;
            }
          }
          if (existing_index >= 0) {
            playlist_index_ = existing_index;
          } else {
            playlist_.insert(playlist_.begin(), u);
            playlist_index_ = 0;
            inserted = true;
          }
        }

        ok = open_media_internal(u, true, err);
        if (!ok) {
          std::lock_guard<std::mutex> lock(state_mu_);
          if (inserted) {
            if (!playlist_.empty() && playlist_.front() == u) {
              playlist_.erase(playlist_.begin());
            } else {
              for (auto it = playlist_.begin(); it != playlist_.end(); ++it) {
                if (*it == u) {
                  playlist_.erase(it);
                  break;
                }
              }
            }
          }
          if (playlist_.empty()) {
            playlist_index_ = -1;
          } else if (previous_index < 0) {
            playlist_index_ = -1;
          } else if (previous_index >= static_cast<int>(playlist_.size())) {
            playlist_index_ = static_cast<int>(playlist_.size()) - 1;
          } else {
            playlist_index_ = previous_index;
          }
        }
      }
    }
  } else if (f == "volume") {
    ok = cmd_set_volume(json{{"volume", value}}, err);
  } else if (f == "position") {
    ok = cmd_seek(json{{"position", value}}, err);
  } else if (f == "loop") {
    if (!value.is_boolean()) {
      err = "loop must be boolean";
      ok = false;
    } else {
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        loop_ = value.get<bool>();
      }
      ok = true;
    }
  } else if (f == "videoOutputMaxWidth" || f == "videoOutputMaxHeight") {
    double number = 0.0;
    if (!parse_finite_number(value, f.c_str(), number, err) || number < 0.0 ||
        number > static_cast<double>(std::numeric_limits<std::uint32_t>::max()) || std::floor(number) != number) {
      if (err.empty())
        err = "value must be a non-negative 32-bit integer";
    } else {
      const auto v = static_cast<std::uint32_t>(number);
      if (f == "videoOutputMaxWidth")
        cfg_.video_output_max_width = v;
      else
        cfg_.video_output_max_height = v;
      if (player_)
        player_->setVideoOutputMaxSize(cfg_.video_output_max_width, cfg_.video_output_max_height);
      ok = true;
    }
  } else if (f == "videoOutputMaxFps") {
    double fps = 0.0;
    if (!parse_finite_number(value, f.c_str(), fps, err) || fps < 0.0) {
      if (err.empty())
        err = "value must be >= 0";
    } else {
      cfg_.video_output_max_fps = fps;
      if (player_)
        player_->setVideoOutputMaxFps(cfg_.video_output_max_fps);
      ok = true;
    }
  } else if (f == "authMode") {
    if (!value.is_string()) {
      err = "authMode must be a string";
      ok = false;
    } else {
      std::string mode;
      if (!parse_auth_mode(value.get<std::string>(), mode)) {
        err = "authMode must be one of: none|browser|cookiesFile";
        ok = false;
      } else {
        std::lock_guard<std::mutex> lock(state_mu_);
        ok = apply_auth_setting_locked(auth_mode_, mode, err);
        if (!ok) {
          last_error_ = err;
        } else {
          last_error_.clear();
        }
      }
    }
  } else if (f == "authBrowser") {
    if (!value.is_string()) {
      err = "authBrowser must be a string";
      ok = false;
    } else {
      const std::string browser = lowercase_ascii(trim_copy(value.get<std::string>()));
      if (!is_supported_auth_browser(browser)) {
        err = "authBrowser must be one of: chrome|chromium|edge|firefox|safari";
        ok = false;
      } else {
        std::lock_guard<std::mutex> lock(state_mu_);
        ok = apply_auth_setting_locked(auth_browser_, browser, err);
        if (!ok) {
          last_error_ = err;
        } else {
          last_error_.clear();
        }
      }
    }
  } else if (f == "authBrowserProfile") {
    if (!value.is_string()) {
      err = "authBrowserProfile must be a string";
      ok = false;
    } else {
      const std::string profile = trim_copy(value.get<std::string>());
      if (!is_profile_value_safe(profile)) {
        err = "authBrowserProfile contains unsupported characters";
        ok = false;
      } else {
        std::lock_guard<std::mutex> lock(state_mu_);
        ok = apply_auth_setting_locked(auth_browser_profile_, profile, err);
        if (!ok) {
          last_error_ = err;
        } else {
          last_error_.clear();
        }
      }
    }
  } else if (f == "authCookiesFile") {
    if (!value.is_string()) {
      err = "authCookiesFile must be a string";
      ok = false;
    } else {
      const std::string cookies_file = trim_copy(value.get<std::string>());
      std::lock_guard<std::mutex> lock(state_mu_);
      ok = apply_auth_setting_locked(auth_cookies_file_, cookies_file, err);
      if (!ok) {
        last_error_ = err;
      } else {
        last_error_.clear();
      }
    }
  } else if (f == "openxrMode") {
    if (!value.is_string()) {
      err = "openxrMode must be a string";
      ok = false;
    } else {
      std::string mode;
      if (!parse_openxr_mode(value.get<std::string>(), mode)) {
        err = "openxrMode must be one of: off|on|auto";
        ok = false;
      } else {
        std::lock_guard<std::mutex> lock(state_mu_);
        openxr_mode_ = mode;
        ok = true;
      }
    }
  } else if (f == "openxrMirrorWindow") {
    if (!value.is_boolean()) {
      err = "openxrMirrorWindow must be boolean";
      ok = false;
    } else {
      const bool mirror = value.get<bool>();
      openxr_mirror_window_.store(mirror, std::memory_order_release);
      ok = true;
    }
  } else {
    error_code = "UNKNOWN_FIELD";
    error_message = "unknown state field";
    return false;
  }

  if (!ok) {
    error_code = "INVALID_VALUE";
    error_message = err.empty() ? "state rejected" : err;
    return false;
  }

  gui_state_dirty_.store(true, std::memory_order_release);

  // Avoid writing high-frequency values (e.g. position) into the KV bucket.
  if (f == "position") {
    return true;
  }

  json write_value = value;
  if (f == "volume") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = volume_;
  } else if (f == "mediaUrl") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = media_url_;
  } else if (f == "loop") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = loop_;
  } else if (f == "videoOutputMaxWidth") {
    write_value = cfg_.video_output_max_width;
  } else if (f == "videoOutputMaxHeight") {
    write_value = cfg_.video_output_max_height;
  } else if (f == "videoOutputMaxFps") {
    write_value = cfg_.video_output_max_fps;
  } else if (f == "authMode") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = auth_mode_;
  } else if (f == "authBrowser") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = auth_browser_;
  } else if (f == "authBrowserProfile") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = auth_browser_profile_;
  } else if (f == "authCookiesFile") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = auth_cookies_file_;
  } else if (f == "openxrMode") {
    std::lock_guard<std::mutex> lock(state_mu_);
    write_value = openxr_mode_;
  } else if (f == "openxrMirrorWindow") {
    write_value = openxr_mirror_window_.load(std::memory_order_acquire);
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = published_state_.find(f);
    if (it == published_state_.end() || it->second != write_value) {
      if (bus_ && !is_sensitive_auth_field(f)) {
        if (!bus_->publish_state(cfg_.service_id, f, write_value, "endpoint", meta)) {
          error_code = "INTERNAL";
          error_message = "state persistence failed";
          return false;
        }
      }
      published_state_[f] = write_value;
    }
  }
  return true;
}

bool ImPlayerService::on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta,
                                      std::string& error_code, std::string& error_message) {
  // ServiceBus has already reconciled rungraph `stateValues` into retained state and queued
  // local state delivery onto the service main thread. Keep this hook lightweight so deploy
  // finalization cannot block on mpv/SDL/player work.
  error_code.clear();
  error_message.clear();

  try {
    if (!graph_obj.is_object() || !graph_obj.contains("nodes") || !graph_obj["nodes"].is_array()) {
      return true;
    }

    const auto nodes = graph_obj["nodes"];
    nlohmann::json service_node;
    for (const auto& n : nodes) {
      if (!n.is_object())
        continue;
      const std::string nid = n.value("nodeId", "");
      if (nid != cfg_.service_id)
        continue;

      // Service node snapshot has no operatorClass.
      bool is_service_snapshot = true;
      if (n.contains("operatorClass") && !n["operatorClass"].is_null()) {
        const auto& operator_class = n["operatorClass"];
        const std::string oc = operator_class.is_string() ? operator_class.get<std::string>() : "";
        if (!oc.empty())
          is_service_snapshot = false;
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

    nlohmann::json meta2 = meta;
    if (!meta2.is_object())
      meta2 = nlohmann::json::object();
    meta2["via"] = "rungraph";
    meta2["graphId"] = graph_obj.value("graphId", "");

    const auto& values = service_node["stateValues"];
    bool has_deferred_state = false;
    for (auto it = values.begin(); it != values.end(); ++it) {
      const std::string field = it.key();
      if (field.empty())
        continue;

      if (field != "mediaUrl" && field != "volume" && field != "videoOutputMaxWidth" &&
          field != "videoOutputMaxHeight" && field != "videoOutputMaxFps" && field != "authMode" &&
          field != "authBrowser") {
        continue;
      }
      has_deferred_state = true;
    }

    if (has_deferred_state && bus_) {
      bus_->post_main_thread([this]() { publish_rungraph_reconcile_snapshot(); });
    }
  } catch (const std::exception& exc) {
    spdlog::warn("implayer set_rungraph state reconcile failed serviceId={}: {}", cfg_.service_id, exc.what());
  } catch (...) {
    // Deploy should not fail because of a local parse issue.
    spdlog::warn("implayer set_rungraph state reconcile failed serviceId={}: unknown error", cfg_.service_id);
  }

  return true;
}

void ImPlayerService::publish_rungraph_reconcile_snapshot() {
  publish_static_state();
  publish_dynamic_state();
}

bool ImPlayerService::on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                                 nlohmann::json& result, std::string& error_code, std::string& error_message) {
  std::string err;
  bool ok = false;

  if (call == "open")
    ok = cmd_open(args, err);
  else if (call == "play")
    ok = cmd_play(err);
  else if (call == "pause")
    ok = cmd_pause(err);
  else if (call == "stop") {
    // Stopping mpv can release OpenGL resources; its context belongs to the SDL thread.
    auto completion = std::make_shared<std::promise<std::pair<bool, std::string>>>();
    auto future = completion->get_future();
    bus_->post_main_thread([this, completion]() {
      std::string stop_error;
      const bool stopped = cmd_stop(stop_error);
      completion->set_value({stopped, std::move(stop_error)});
    });
    if (future.wait_for(std::chrono::milliseconds(1500)) != std::future_status::ready) {
      err = "timed out waiting for the playback thread";
    } else {
      auto [stopped, stop_error] = future.get();
      ok = stopped;
      err = std::move(stop_error);
    }
  }
  else if (call == "next")
    ok = cmd_next(err);
  else if (call == "previous")
    ok = cmd_previous(err);
  else if (call == "seek")
    ok = cmd_seek(args, err);
  else if (call == "setVolume")
    ok = cmd_set_volume(args, err);
  else {
    error_code = "UNKNOWN_CALL";
    error_message = "unknown call: " + call;
    return false;
  }

  if (!ok) {
    error_code = "INTERNAL";
    error_message = err.empty() ? "command failed" : err;
    return false;
  }

  (void)meta;
  gui_state_dirty_.store(true, std::memory_order_release);
  result = json::object();
  return true;
}

void ImPlayerService::playlist_add(const std::vector<std::string>& items, bool play_if_idle) {
  if (items.empty())
    return;

  std::string url_to_open;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const std::string current_url = normalize_url(media_url_);

    std::unordered_set<std::string> seen;
    std::vector<std::string> next_playlist;
    next_playlist.reserve(playlist_.size() + items.size());

    for (const auto& existing : playlist_) {
      const std::string u = normalize_url(existing);
      if (u.empty())
        continue;
      if (!seen.insert(u).second)
        continue;
      next_playlist.push_back(u);
    }

    const bool empty_before = next_playlist.empty();
    for (const auto& raw : items) {
      const std::string u = normalize_url(raw);
      if (u.empty())
        continue;
      if (!seen.insert(u).second)
        continue;
      next_playlist.push_back(u);
    }

    playlist_ = std::move(next_playlist);
    if (playlist_.empty())
      return;

    if (!current_url.empty()) {
      for (std::size_t i = 0; i < playlist_.size(); ++i) {
        if (playlist_[i] == current_url) {
          playlist_index_ = static_cast<int>(i);
          break;
        }
      }
    }
    if (empty_before) {
      playlist_index_ = 0;
      url_to_open = playlist_[0];
    } else if (play_if_idle && playlist_index_ < 0) {
      playlist_index_ = 0;
      url_to_open = playlist_[0];
    } else if (play_if_idle && playlist_index_ >= static_cast<int>(playlist_.size())) {
      playlist_index_ = 0;
      url_to_open = playlist_[0];
    }
  }

  if (!url_to_open.empty()) {
    std::string err;
    if (play_if_idle) {
      set_playback_intent(PlaybackIntent::Playing);
    }
    (void)open_media_internal(url_to_open, true, err);
  }
}

void ImPlayerService::playlist_play_index(int index) {
  std::string url;
  int previous_index = -1;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (index < 0 || index >= static_cast<int>(playlist_.size()))
      return;
    previous_index = playlist_index_;
    playlist_index_ = index;
    url = playlist_[static_cast<std::size_t>(playlist_index_)];
  }
  std::string err;
  if (!open_media_internal(url, true, err)) {
    std::lock_guard<std::mutex> lock(state_mu_);
    playlist_index_ = previous_index;
    last_error_ = "Playlist: " + err;
  }
}

void ImPlayerService::playlist_remove_index(int index) {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (index < 0 || index >= static_cast<int>(playlist_.size())) {
    return;
  }

  const std::size_t i = static_cast<std::size_t>(index);
  playlist_.erase(playlist_.begin() + static_cast<std::ptrdiff_t>(i));
  if (playlist_.empty()) {
    playlist_index_ = -1;
    return;
  }

  if (playlist_index_ == index) {
    // The currently-highlighted item was removed. Keep playback state unchanged and
    // force the selection to "none" to avoid highlighting the wrong item.
    playlist_index_ = -1;
    return;
  }
  if (playlist_index_ > index) {
    playlist_index_ -= 1;
  }
}

void ImPlayerService::playlist_clear() {
  std::lock_guard<std::mutex> lock(state_mu_);
  playlist_.clear();
  playlist_index_ = -1;
}

void ImPlayerService::playlist_next() {
  std::string url;
  bool loop = false;
  int previous_index = -1;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (playlist_.empty())
      return;
    previous_index = playlist_index_;
    const int next = playlist_index_ < 0 ? 0 : playlist_index_ + 1;
    loop = loop_;
    if (next >= static_cast<int>(playlist_.size())) {
      if (!loop)
        return;
      playlist_index_ = 0;
      url = playlist_[0];
    } else {
      playlist_index_ = next;
      url = playlist_[static_cast<std::size_t>(playlist_index_)];
    }
  }
  std::string err;
  if (!open_media_internal(url, true, err)) {
    std::lock_guard<std::mutex> lock(state_mu_);
    playlist_index_ = previous_index;
    last_error_ = "Next: " + err;
  }
}

void ImPlayerService::playlist_prev() {
  std::string url;
  int previous_index = -1;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (playlist_.empty())
      return;
    if (playlist_index_ <= 0)
      return;
    previous_index = playlist_index_;
    playlist_index_ -= 1;
    url = playlist_[static_cast<std::size_t>(playlist_index_)];
  }
  std::string err;
  if (!open_media_internal(url, true, err)) {
    std::lock_guard<std::mutex> lock(state_mu_);
    playlist_index_ = previous_index;
    last_error_ = "Previous: " + err;
  }
}

bool ImPlayerService::open_media_internal(const std::string& url, bool keep_playlist, std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  const std::string u = normalize_url(url);
  if (u.empty()) {
    err = "missing url";
    return false;
  }
  const PlaybackIntent intent = playback_intent();
  const bool wants_loaded_media = playback_intent_wants_loaded_media(intent);
  const bool wants_playback = playback_intent_wants_playback(intent);

  if (!wants_loaded_media) {
    eof_reached_.store(false, std::memory_order_release);
    media_finished_.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      media_url_ = u;
      last_error_.clear();
      if (!keep_playlist) {
        playlist_.clear();
        playlist_.push_back(u);
        playlist_index_ = 0;
      }
    }
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!media_url_.empty() && media_url_ == u) {
      last_error_.clear();
      if (!keep_playlist) {
        playlist_.clear();
        playlist_.push_back(u);
        playlist_index_ = 0;
      }
      // If the user previously hit Stop, mpv likely unloaded the file. Reload it.
      if (stopped_.load(std::memory_order_acquire)) {
        if (!playback_intent_should_reload_stopped_media(intent))
          return true;
      } else {
        if (eof_reached_.load(std::memory_order_acquire)) {
          eof_reached_.store(false, std::memory_order_release);
          media_finished_.store(false, std::memory_order_release);
          player_->seek(0.0);
        }
        if (wants_playback) {
          if (!player_->play()) {
            err = "mpv could not start playback";
            return false;
          }
          playing_.store(true, std::memory_order_release);
        } else {
          player_->pause();
          playing_.store(false, std::memory_order_release);
        }
        return true;
      }
    }
  }

  YtdlpAuthOptions ytdlp_auth;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!apply_auth_options_locked(err)) {
      last_error_ = err;
      return false;
    }
    ytdlp_auth.mode = auth_mode_;
    ytdlp_auth.browser = auth_browser_;
    ytdlp_auth.browser_profile = auth_browser_profile_;
    ytdlp_auth.cookies_file = auth_cookies_file_;
  }

  if (!open_player_media(*player_, u, ytdlp_auth, err)) {
    std::lock_guard<std::mutex> lock(state_mu_);
    last_error_ = err;
    return false;
  }
  eof_reached_.store(false, std::memory_order_release);
  stopped_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    media_url_ = u;
    last_error_.clear();
    video_id_ = new_video_id();
    if (!keep_playlist) {
      playlist_.clear();
      playlist_.push_back(u);
      playlist_index_ = 0;
    }

    vr_sbs_eye_ = 0;
    vr_yaw_deg_ = 0.0f;
    vr_pitch_deg_ = 0.0f;
    vr_fov_deg_ = 90.0f;
    vr_dragging_ = false;
  }
  if (wants_playback) {
    if (!player_->play()) {
      err = "mpv could not start playback";
      return false;
    }
    playing_.store(true, std::memory_order_release);
  } else {
    player_->pause();
    playing_.store(false, std::memory_order_release);
  }

  return true;
}

bool ImPlayerService::cmd_open(const nlohmann::json& args, std::string& err) {
  std::string url;
  if (args.is_object()) {
    if (args.contains("url") && args["url"].is_string())
      url = args["url"].get<std::string>();
    if (url.empty() && args.contains("mediaUrl") && args["mediaUrl"].is_string())
      url = args["mediaUrl"].get<std::string>();
  }
  url = normalize_url(std::move(url));
  if (url.empty()) {
    err = "missing url";
    return false;
  }
  const PlaybackIntent previous_intent = playback_intent();
  set_playback_intent(PlaybackIntent::Playing);
  if (open_media_internal(url, false, err))
    return true;
  set_playback_intent(previous_intent);
  return false;
}

bool ImPlayerService::cmd_play(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (media_url_.empty()) {
      err = "no media loaded";
      return false;
    }
  }
  const PlaybackIntent previous_intent = playback_intent();
  set_playback_intent(PlaybackIntent::Playing);
  const bool reloaded_from_stop = stopped_.load(std::memory_order_acquire);
  if (reloaded_from_stop) {
    std::string url;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      url = media_url_;
    }
    if (url.empty()) {
      err = "no media loaded";
      set_playback_intent(previous_intent);
      return false;
    }
    YtdlpAuthOptions ytdlp_auth;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (!apply_auth_options_locked(err)) {
        last_error_ = err;
        set_playback_intent(previous_intent);
        return false;
      }
      ytdlp_auth.mode = auth_mode_;
      ytdlp_auth.browser = auth_browser_;
      ytdlp_auth.browser_profile = auth_browser_profile_;
      ytdlp_auth.cookies_file = auth_cookies_file_;
    }
    if (!open_player_media(*player_, url, ytdlp_auth, err)) {
      set_playback_intent(previous_intent);
      return false;
    }
    eof_reached_.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      last_error_.clear();
    }
    player_->seek(0.0);
  }
  if (eof_reached_.load(std::memory_order_acquire)) {
    eof_reached_.store(false, std::memory_order_release);
    media_finished_.store(false, std::memory_order_release);
    player_->seek(0.0);
  }
  if (!player_->play()) {
    err = "mpv could not start playback";
    set_playback_intent(previous_intent);
    return false;
  }
  if (reloaded_from_stop) {
    stopped_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(state_mu_);
    video_id_ = new_video_id();
  }
  playing_.store(true, std::memory_order_release);
  return true;
}

bool ImPlayerService::cmd_pause(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  set_playback_intent(PlaybackIntent::Paused);
  player_->pause();
  playing_.store(false, std::memory_order_release);
  return true;
}

bool ImPlayerService::cmd_stop(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  set_playback_intent(PlaybackIntent::Stopped);
  player_->stop();
  player_->resetPlaybackState();
  eof_reached_.store(false, std::memory_order_release);
  media_finished_.store(false, std::memory_order_release);
  stopped_.store(true, std::memory_order_release);
  clear_video_requested_.store(true, std::memory_order_release);
  playing_.store(false, std::memory_order_release);
  position_seconds_.store(0.0, std::memory_order_release);
  duration_seconds_.store(0.0, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    last_error_.clear();
    video_id_.clear();
  }
  return true;
}

bool ImPlayerService::cmd_next(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  playlist_next();
  return true;
}

bool ImPlayerService::cmd_previous(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  playlist_prev();
  return true;
}

bool ImPlayerService::cmd_seek(const nlohmann::json& args, std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  if (!args.is_object() || !args.contains("position")) {
    err = "missing position";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (media_url_.empty()) {
      err = "no media loaded";
      return false;
    }
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (media_url_.empty()) {
      err = "no media loaded";
      return false;
    }
  }
  double pos = 0.0;
  if (!parse_finite_number(args["position"], "position", pos, err))
    return false;
  if (pos < 0.0) {
    err = "position must be >= 0";
    return false;
  }
  const PlaybackIntent intent = playback_intent();
  const bool wants_loaded_media = playback_intent_wants_loaded_media(intent);
  const bool wants_playback = playback_intent_wants_playback(intent);
  if (!wants_loaded_media) {
    set_playback_intent(PlaybackIntent::Paused);
    if (cmd_seek(args, err))
      return true;
    set_playback_intent(intent);
    return false;
  }
  const bool was_stopped = stopped_.load(std::memory_order_acquire);
  if (eof_reached_.load(std::memory_order_acquire) || was_stopped) {
    std::string url;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      url = media_url_;
    }
    if (url.empty()) {
      err = "no media loaded";
      return false;
    }
    YtdlpAuthOptions ytdlp_auth;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (!apply_auth_options_locked(err)) {
        last_error_ = err;
        return false;
      }
      ytdlp_auth.mode = auth_mode_;
      ytdlp_auth.browser = auth_browser_;
      ytdlp_auth.browser_profile = auth_browser_profile_;
      ytdlp_auth.cookies_file = auth_cookies_file_;
    }
    if (!open_player_media(*player_, url, ytdlp_auth, err)) {
      std::lock_guard<std::mutex> lock(state_mu_);
      last_error_ = err;
      return false;
    }
    eof_reached_.store(false, std::memory_order_release);
    stopped_.store(false, std::memory_order_release);
    if (was_stopped) {
      std::lock_guard<std::mutex> lock(state_mu_);
      last_error_.clear();
      video_id_ = new_video_id();
    }
  }
  player_->seek(pos);
  if (wants_playback) {
    if (!player_->play()) {
      err = "mpv could not resume after seeking";
      return false;
    }
    playing_.store(true, std::memory_order_release);
  } else {
    player_->pause();
    playing_.store(false, std::memory_order_release);
  }
  return true;
}

bool ImPlayerService::cmd_set_volume(const nlohmann::json& args, std::string& err) {
  if (!args.is_object() || !args.contains("volume")) {
    err = "missing volume";
    return false;
  }
  double vol = 1.0;
  if (!parse_finite_number(args["volume"], "volume", vol, err))
    return false;
  if (vol < 0.0 || vol > 1.0) {
    err = "volume must be between 0 and 1";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    volume_ = vol;
  }
  if (player_)
    player_->setVolume(vol);
  return true;
}

void ImPlayerService::report_gui_result(const char* action, bool success, const std::string& error) {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (success) {
    last_error_.clear();
  } else {
    last_error_ = std::string(action) + ": " + (error.empty() ? "operation failed" : error);
  }
}

bool ImPlayerService::apply_auth_setting_locked(std::string& setting, const std::string& next, std::string& err) {
  const std::string previous = setting;
  setting = next;
  if (apply_auth_options_locked(err))
    return true;
  setting = previous;
  std::string restore_error;
  if (!apply_auth_options_locked(restore_error))
    spdlog::error("failed to restore previous auth options: {}", restore_error);
  return false;
}

bool ImPlayerService::apply_auth_options_locked(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  if (!is_supported_auth_mode(auth_mode_)) {
    err = "authMode must be one of: none|browser|cookiesFile";
    return false;
  }

  if (auth_mode_ == "none") {
    if (!player_->setYtdlRawOptions("")) {
      err = "failed to clear ytdl-raw-options";
      return false;
    }
    if (!player_->setCookiesFile("")) {
      err = "failed to clear cookies-file";
      return false;
    }
    return true;
  }

  if (auth_mode_ == "browser") {
    if (!is_supported_auth_browser(auth_browser_)) {
      err = "authBrowser must be one of: chrome|chromium|edge|firefox|safari";
      return false;
    }
    if (!is_profile_value_safe(auth_browser_profile_)) {
      err = "authBrowserProfile contains unsupported characters";
      return false;
    }
    if (!player_->setCookiesFile("")) {
      err = "failed to clear cookies-file";
      return false;
    }
    // Newer libmpv builds can omit ytdl_hook entirely. Browser cookies are
    // passed to the explicit yt-dlp resolver instead of ytdl-raw-options.
    return true;
  }

  // auth_mode_ == "cookiesFile"
  const std::string file = trim_copy(auth_cookies_file_);
  if (file.empty()) {
    err = "authCookiesFile must be a non-empty path when authMode=cookiesFile";
    return false;
  }
  if (!std::filesystem::exists(file)) {
    err = "authCookiesFile does not exist: " + file;
    return false;
  }
  (void)player_->setYtdlRawOptions("");
  if (!player_->setCookiesFile(file)) {
    err = "failed to set cookies-file";
    return false;
  }
  return true;
}

void ImPlayerService::publish_static_state() {
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
    want("videoFormat", "bgra32");
    want("videoFrameSchemaVersion", 1);
    want("loop", loop_);
    want("videoOutputMaxWidth", cfg_.video_output_max_width);
    want("videoOutputMaxHeight", cfg_.video_output_max_height);
    want("videoOutputMaxFps", cfg_.video_output_max_fps);
    want("authMode", auth_mode_);
    want("authBrowser", auth_browser_);
    want("openxrMode", openxr_mode_);
    want("openxrMirrorWindow", openxr_mirror_window_.load(std::memory_order_acquire));
  }
  for (const auto& [field, v] : updates) {
    if (bus_) {
      (void)bus_->publish_state(cfg_.service_id, field, v, "runtime", meta);
    }
  }
}

void ImPlayerService::publish_dynamic_state() {
  const json meta = json{{"via", "periodic"}};

  const bool playing = playing_.load(std::memory_order_relaxed);
  const double dur = duration_seconds_.load(std::memory_order_relaxed);
  const unsigned decoded_w = player_ ? player_->videoWidth() : 0;
  const unsigned decoded_h = player_ ? player_->videoHeight() : 0;

  std::vector<std::pair<std::string, json>> updates;
  bool error_changed = false;
  std::string error_message;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const double vol = volume_;
    const std::string url = media_url_;
    const std::string err = last_error_;
    const bool loop = loop_;

    auto want = [&](const std::string& field, const json& v) {
      auto it = published_state_.find(field);
      if (it != published_state_.end() && it->second == v)
        return;
      published_state_[field] = v;
      updates.emplace_back(field, v);
    };

    want("playing", playing);
    want("duration", dur);
    want("volume", vol);
    want("loop", loop);
    want("mediaUrl", url);
    if (published_error_message_ != err) {
      published_error_message_ = err;
      error_message = err;
      error_changed = true;
    }

    want("decodedWidth", static_cast<std::int64_t>(decoded_w));
    want("decodedHeight", static_cast<std::int64_t>(decoded_h));

    if (frame_sink_) {
      want("videoWidth", frame_sink_->outputWidth());
      want("videoHeight", frame_sink_->outputHeight());
      want("videoPitch", frame_sink_->outputPitch());
    }
  }

  if (bus_ && error_changed) {
    if (error_message.empty()) {
      bus_->clear_error(cfg_.service_id);
    } else {
      bus_->report_error(cfg_.service_id, "IMPLAYER_ERROR", error_message, "error",
                         cfg_.service_id + ":" + error_message);
    }
  }

  for (const auto& [field, v] : updates) {
    if (bus_) {
      (void)bus_->publish_state(cfg_.service_id, field, v, "runtime", meta);
    }
  }
}

json ImPlayerService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.implayer";
  service["label"] = "IM Player";
  service["version"] = "0.0.1";
  service["description"] = "C++ MPV-based player service with Zenoh latest-frame video output.";
  service["stateFields"] = json::array({
      state_field("loop", schema_boolean(), "rw", "Loop", "Repeat playlist when reaching EOF.", false),
      state_field("mediaUrl", schema_string(), "rw", "Media URL",
                  "URI or local file path to open. Cleared when exporting publish JSON.", true, "", true),
      state_field("openxrMode", schema_string_enum({"off", "on", "auto"}), "rw", "OpenXR Mode",
                  "PCVR output: off|on|auto (auto retries when headset/runtime becomes available).", true),
      state_field("openxrMirrorWindow", schema_boolean(), "rw", "OpenXR Mirror",
                  "When OpenXR is active, also present to the SDL mirror window.", true),
      state_field("volume", schema_number(1.0, 0.0, 1.0), "rw", "Volume", "", true, "slider"),
      state_field("playing", schema_boolean(), "ro", "Playing", "Playback state.", false),
      state_field("duration", schema_number(), "ro", "Duration", "Duration (seconds).", true),
      state_field("videoFormat", schema_string_enum({"bgra32", "bgr24", "flow2_f16", "scalar1_f32"}), "ro",
                  "Video Format", "Frame payload format.", false),
      state_field("videoFrameSchemaVersion", schema_integer(), "ro", "Video Schema", "Frame schema version.", false),
      state_field("videoOutputMaxWidth", schema_integer(), "rw", "Output Max Width", "Downsample limit (0 = auto).",
                  false),
      state_field("videoOutputMaxHeight", schema_integer(), "rw", "Output Max Height", "Downsample limit (0 = auto).",
                  false),
      state_field("videoOutputMaxFps", schema_number(), "rw", "Output Max FPS",
                  "Frame export rate limit (0 = unlimited).", false),
      state_field("authMode", schema_string_enum({"none", "browser", "cookiesFile"}), "rw", "Auth Mode",
                  "Cookie auth mode: none|browser|cookiesFile (default: none).", false),
      state_field("authBrowser", schema_string_enum({"chrome", "chromium", "edge", "firefox", "safari"}), "rw",
                  "Auth Browser",
                  "Browser name for authMode=browser: chrome|chromium|edge|firefox|safari.", false),
      state_field("authBrowserProfile", schema_string(), "rw", "Auth Browser Profile",
                  "Optional browser profile for authMode=browser. Local-only path-like metadata; cleared when exporting publish JSON.",
                  false, "", true),
      state_field("authCookiesFile", schema_string(), "rw", "Auth Cookies File",
                  "cookies.txt path for authMode=cookiesFile. Local-only file path; cleared when exporting publish JSON.",
                  false, "", true),
      state_field("decodedWidth", schema_integer(), "ro", "Decoded Width",
                  "Decoded/source video width (on-screen uses this).", false),
      state_field("decodedHeight", schema_integer(), "ro", "Decoded Height",
                  "Decoded/source video height (on-screen uses this).", false),
      state_field("videoWidth", schema_integer(), "ro", "Width", "Width of the video frame.", false),
      state_field("videoHeight", schema_integer(), "ro", "Height", "Height of the video frame.", false),
      state_field("videoPitch", schema_integer(), "ro", "Pitch", "Pitch of the video frame.", false),
  });

  service["dataOutPorts"] = json::array({
      video_frame_port("video", "Decoded video frame stream."),
      json{{"name", "playback"},
           {"valueSchema", schema_object(json{{"videoId", schema_string()},
                                              {"position", schema_number()},
                                              {"duration", schema_number()},
                                              {"playing", schema_boolean()}},
                                         json::array({"videoId", "position"}))},
           {"description", "Playback state stream (position/duration/playing)."},
           {"required", true},
           {"showOnNode", false}},
  });
  service["commands"] = json::array({
      json{{"name", "open"},
           {"description", "Open a media URL"},
           {"required", true},
           {"showOnNode", true},
           {"params", json::array({json{{"name", "url"}, {"valueSchema", schema_string()}, {"required", true}}})}},
      json{{"name", "play"}, {"description", "Start playback"}, {"required", true}, {"showOnNode", true}},
      json{{"name", "pause"}, {"description", "Pause playback"}, {"required", true}, {"showOnNode", true}},
      json{{"name", "stop"}, {"description", "Stop playback"}, {"required", true}, {"showOnNode", true}},
      json{{"name", "next"},
           {"description", "Advance to the next playlist item"},
           {"required", true},
           {"showOnNode", true}},
      json{{"name", "previous"},
           {"description", "Return to the previous playlist item"},
           {"required", true},
           {"showOnNode", true}},
      json{{"name", "seek"},
           {"description", "Seek"},
           {"required", true},
           {"params", json::array({json{{"name", "position"}, {"valueSchema", schema_number()}, {"required", true}}})}},
      json{{"name", "setVolume"},
           {"description", "Set volume"},
           {"required", true},
           {"params", json::array({json{{"name", "volume"}, {"valueSchema", schema_number()}, {"required", true}}})}},
  });

  json out;
  out["service"] = service;
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::implayer
