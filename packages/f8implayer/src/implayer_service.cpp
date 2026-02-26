#include "implayer_service.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "f8cppsdk/data_bus.h"
#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/shm/video.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/video_shared_memory_sink.h"
#include "implayer_gui.h"
#include "mpv_player.h"
#include "sdl_video_window.h"

namespace f8::implayer {

using json = nlohmann::json;

namespace {

json schema_string() {
  return json{{"type", "string"}};
}
json schema_string_enum(std::initializer_list<const char*> items) {
  json s = schema_string();
  s["enum"] = json::array();
  for (const char* it : items) {
    if (it && *it)
      s["enum"].push_back(it);
  }
  return s;
}
json schema_number() {
  return json{{"type", "number"}};
}
json schema_number(double minimum, double maximum) {
  json s{{"type", "number"}};
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
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
                 std::string description = {}, bool show_on_node = false, std::string ui_control = {}) {
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
  if (!ui_control.empty())
    sf["uiControl"] = std::move(ui_control);
  return sf;
}

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

std::string browser_cookie_option_value(const std::string& browser, const std::string& profile) {
  if (profile.empty())
    return browser;
  return browser + ":" + profile;
}

float normalize_yaw_deg(float yaw_deg) {
  float v = std::fmod(yaw_deg, 360.0f);
  if (v > 180.0f)
    v -= 360.0f;
  if (v < -180.0f)
    v += 360.0f;
  return v;
}

SdlVideoWindow::ProjectionMode detect_projection_from_name(const std::string& url) {
  const std::string lower = lowercase_ascii(url);
  const bool has_360 = (lower.find("360") != std::string::npos) || (lower.find("vr") != std::string::npos) ||
                       (lower.find("equirect") != std::string::npos);
  if (!has_360) {
    return SdlVideoWindow::ProjectionMode::Flat2D;
  }
  const bool has_sbs = (lower.find("sbs") != std::string::npos) || (lower.find("sidebyside") != std::string::npos) ||
                       (lower.find("side-by-side") != std::string::npos) || (lower.find("_lr") != std::string::npos) ||
                       (lower.find("-lr") != std::string::npos);
  if (has_sbs) {
    return SdlVideoWindow::ProjectionMode::EquirectSbs;
  }
  return SdlVideoWindow::ProjectionMode::EquirectMono;
}

SdlVideoWindow::ProjectionMode detect_projection_from_ratio(unsigned width, unsigned height) {
  if (width == 0 || height == 0) {
    return SdlVideoWindow::ProjectionMode::Flat2D;
  }
  const double ratio = static_cast<double>(width) / static_cast<double>(height);
  if (std::abs(ratio - 1.0) <= 0.08) {
    return SdlVideoWindow::ProjectionMode::EquirectSbs;
  }
  if (std::abs(ratio - 2.0) <= 0.12) {
    return SdlVideoWindow::ProjectionMode::EquirectMono;
  }
  return SdlVideoWindow::ProjectionMode::Flat2D;
}

MpvPlayer::ShmViewMode shm_view_mode_for_vr(SdlVideoWindow::ProjectionMode mode, int sbs_eye) {
  if (mode == SdlVideoWindow::ProjectionMode::EquirectSbs) {
    return sbs_eye == 0 ? MpvPlayer::ShmViewMode::SbsLeft : MpvPlayer::ShmViewMode::SbsRight;
  }
  return MpvPlayer::ShmViewMode::FullFrame;
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

  shm_ = std::make_shared<VideoSharedMemorySink>();
  const auto shm_name = f8::cppsdk::shm::video_shm_name(cfg_.service_id);
  if (!shm_->initialize(shm_name, cfg_.video_shm_bytes, cfg_.video_shm_slots)) {
    spdlog::error("failed to initialize video shm sink name={} bytes={} slots={}", shm_name, cfg_.video_shm_bytes,
                  cfg_.video_shm_slots);
    return false;
  }

  SdlVideoWindow::Config wcfg;
  wcfg.title = "f8implayer - " + cfg_.service_id;
  wcfg.width = cfg_.window_width;
  wcfg.height = cfg_.window_height;
  wcfg.resizable = cfg_.window_resizable;
  wcfg.vsync = cfg_.window_vsync;
  window_ = std::make_unique<SdlVideoWindow>(wcfg);
  if (!window_->start()) {
    spdlog::error("failed to start SDL video window");
    return false;
  }
  if (!window_->makeCurrent()) {
    spdlog::error("failed to activate SDL GL context");
    return false;
  }

  gui_ = std::make_unique<ImPlayerGui>();
  if (!gui_->start(window_->sdlWindow(), window_->glContext())) {
    spdlog::error("failed to initialize ImGui overlay");
    return false;
  }

  MpvPlayer::VideoConfig vcfg;
  vcfg.offline = false;
  vcfg.videoShmMaxWidth = cfg_.video_shm_max_width;
  vcfg.videoShmMaxHeight = cfg_.video_shm_max_height;
  vcfg.videoShmMaxFps = cfg_.video_shm_max_fps;

  try {
    player_ = std::make_unique<MpvPlayer>(
        vcfg,
        [this](double pos, double dur) {
          position_seconds_.store(pos, std::memory_order_relaxed);
          duration_seconds_.store(dur, std::memory_order_relaxed);
        },
        [this](bool playing) { playing_.store(playing, std::memory_order_relaxed); },
        [this]() {
          playing_.store(false, std::memory_order_relaxed);
          media_finished_.store(true, std::memory_order_release);
          eof_reached_.store(true, std::memory_order_release);
        });
  } catch (const std::exception& e) {
    spdlog::error("mpv init failed: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("mpv init failed: unknown error");
    return false;
  }
  player_->setSharedMemorySink(shm_);
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
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_set_state_node(this);
  bus_->add_rungraph_node(this);
  bus_->add_command_node(this);
  if (!bus_->start()) {
    bus_.reset();
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
  spdlog::info("implayer started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void ImPlayerService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  stop_requested_.store(true, std::memory_order_release);

  try {
    if (bus_)
      bus_->stop();
  } catch (...) {}
  bus_.reset();

  if (window_)
    window_->makeCurrent();
  if (player_)
    player_->shutdownGl();
  if (gui_)
    gui_->stop();
  gui_.reset();
  player_.reset();
  window_.reset();
  shm_.reset();
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
      if (vw != 0 && vh != 0 && vr_auto_pending_ratio_ && !vr_manual_override_) {
        const auto ratio_mode = detect_projection_from_ratio(vw, vh);
        if (ratio_mode != SdlVideoWindow::ProjectionMode::Flat2D || !vr_auto_detect_valid_) {
          vr_auto_detect_mode_ = ratio_mode;
          vr_auto_detect_valid_ = true;
          vr_mode_ = ratio_mode;
        }
        vr_auto_pending_ratio_ = false;
      }

      player_->setShmViewMode(shm_view_mode_for_vr(vr_mode_, vr_sbs_eye_));
      const bool updated = player_->renderVideoFrame();
      if (force_present || updated || window_->needsRedraw() || (gui_ && gui_->wantsRepaint())) {
        ImPlayerGui::Callbacks cb;
        cb.open = [this](const std::string& url) {
          std::string err;
          (void)open_media_internal(url, false, err);
        };
        cb.play = [this]() {
          std::string err;
          (void)cmd_play(err);
        };
        cb.pause = [this]() {
          std::string err;
          (void)cmd_pause(err);
        };
        cb.stop = [this]() {
          std::string err;
          (void)cmd_stop(err);
        };
        cb.seek = [this](double pos) {
          std::string err;
          (void)cmd_seek(json{{"position", pos}}, err);
        };
        cb.set_volume = [this](double vol) {
          std::string err;
          (void)cmd_set_volume(json{{"volume", vol}}, err);
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
          mark_vr_manual_override();
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
        {
          std::lock_guard<std::mutex> lock(state_mu_);
          err = last_error_;
          playlist_snapshot = playlist_;
          playlist_index_snapshot = playlist_index_;
          loop_snapshot = loop_;
        }

        const SdlVideoWindow::ViewTransform view{view_zoom_, view_pan_x_, view_pan_y_};
        const SdlVideoWindow::VrViewState vr_view{
            vr_mode_, vr_yaw_deg_, std::clamp(vr_pitch_deg_, -89.0f, 89.0f), std::clamp(vr_fov_deg_, 50.0f, 120.0f),
            vr_sbs_eye_};
        const bool playing = playing_.load(std::memory_order_relaxed);
        window_->present(
            *player_,
            [this, &cb, &err, &playlist_snapshot, playlist_index_snapshot, playing, loop_snapshot]() {
              if (gui_ && player_) {
                gui_->renderOverlay(*player_, cb, err, playlist_snapshot, playlist_index_snapshot, playing,
                                    loop_snapshot, tick_ema_fps_, tick_ema_ms_, vr_mode_, vr_sbs_eye_, vr_yaw_deg_,
                                    vr_pitch_deg_, vr_fov_deg_);
                gui_->clearRepaintFlag();
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

  // High-frequency signals go through the data bus, not KV state.
  if (shm_) {
    const auto frame_id = shm_->frameId();
    if (frame_id != last_frame_id_published_) {
      last_frame_id_published_ = frame_id;
      if (bus_) {
        (void)f8::cppsdk::publish_data(bus_->nats(), cfg_.service_id, cfg_.service_id, "frameId", frame_id, now);
      }
    }
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
      (void)f8::cppsdk::publish_data(bus_->nats(), cfg_.service_id, cfg_.service_id, "playback", evt, now);
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
      if (playing_.load(std::memory_order_relaxed))
        player_->pause();
      else
        (void)player_->play();
    } else if (key == SDLK_LEFT) {
      const double p = position_seconds_.load(std::memory_order_relaxed);
      player_->seek(std::max(0.0, p - 5.0));
    } else if (key == SDLK_RIGHT) {
      const double p = position_seconds_.load(std::memory_order_relaxed);
      player_->seek(p + 5.0);
    } else if (key == SDLK_UP) {
      double v = 1.0;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        v = volume_;
        const double nv = v + 0.05;
        volume_ = nv < 0.0 ? 0.0 : (nv > 1.0 ? 1.0 : nv);
        v = volume_;
      }
      player_->setVolume(v);
    } else if (key == SDLK_DOWN) {
      double v = 1.0;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        v = volume_;
        const double nv = v - 0.05;
        volume_ = nv < 0.0 ? 0.0 : (nv > 1.0 ? 1.0 : nv);
        v = volume_;
      }
      player_->setVolume(v);
    }
  }
}

void ImPlayerService::set_active_local(bool active) {
  active_.store(active, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (player_) {
      if (active)
        player_->play();
      else
        player_->pause();
    }
  }
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
    ok = cmd_open(json{{"url", value}}, err);
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
  } else if (f == "videoShmMaxWidth" || f == "videoShmMaxHeight") {
    if (!value.is_number_integer() && !value.is_number()) {
      err = "value must be a number";
      ok = false;
    } else {
      const auto v = static_cast<std::int64_t>(value.get<double>());
      if (v < 0) {
        err = "value must be >= 0";
        ok = false;
      } else {
        if (f == "videoShmMaxWidth")
          cfg_.video_shm_max_width = static_cast<std::uint32_t>(v);
        if (f == "videoShmMaxHeight")
          cfg_.video_shm_max_height = static_cast<std::uint32_t>(v);
        if (player_)
          player_->setVideoShmMaxSize(cfg_.video_shm_max_width, cfg_.video_shm_max_height);
        ok = true;
      }
    }
  } else if (f == "videoShmMaxFps") {
    if (!value.is_number()) {
      err = "value must be a number";
      ok = false;
    } else {
      const double fps = value.get<double>();
      if (fps < 0.0) {
        err = "value must be >= 0";
        ok = false;
      } else {
        cfg_.video_shm_max_fps = fps;
        if (player_)
          player_->setVideoShmMaxFps(cfg_.video_shm_max_fps);
        ok = true;
      }
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
        auth_mode_ = mode;
        ok = apply_auth_options_locked(err);
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
        auth_browser_ = browser;
        ok = apply_auth_options_locked(err);
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
        auth_browser_profile_ = profile;
        ok = apply_auth_options_locked(err);
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
      auth_cookies_file_ = cookies_file;
      ok = apply_auth_options_locked(err);
      if (!ok) {
        last_error_ = err;
      } else {
        last_error_.clear();
      }
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
  } else if (f == "videoShmMaxWidth") {
    write_value = cfg_.video_shm_max_width;
  } else if (f == "videoShmMaxHeight") {
    write_value = cfg_.video_shm_max_height;
  } else if (f == "videoShmMaxFps") {
    write_value = cfg_.video_shm_max_fps;
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
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = published_state_.find(f);
    if (it == published_state_.end() || it->second != write_value) {
      published_state_[f] = write_value;
      if (bus_ && !is_sensitive_auth_field(f)) {
        f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, f, write_value, "endpoint", meta);
      }
    }
  }
  return true;
}

bool ImPlayerService::on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta,
                                      std::string& error_code, std::string& error_message) {
  // Apply rungraph-provided service node `stateValues` (studio node properties).
  //
  // Studio deploys graphs via the `set_rungraph` endpoint; for python runtimes, the ServiceHost reconciles
  // `stateValues` into KV. This C++ service implements the same behavior for a single service node.
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

    nlohmann::json meta2 = meta;
    if (!meta2.is_object())
      meta2 = nlohmann::json::object();
    meta2["via"] = "rungraph";
    meta2["graphId"] = graph_obj.value("graphId", "");

    const auto& values = service_node["stateValues"];
    for (auto it = values.begin(); it != values.end(); ++it) {
      const std::string field = it.key();
      if (field.empty())
        continue;

      // Only apply writable fields from rungraph (never seed runtime-owned ro fields).
      if (field != "active" && field != "mediaUrl" && field != "volume" && field != "videoShmMaxWidth" &&
          field != "videoShmMaxHeight" && field != "videoShmMaxFps" && field != "authMode" &&
          field != "authBrowser") {
        continue;
      }
      if (field == "active") {
        if (bus_ && it.value().is_boolean()) {
          bus_->set_active_local(it.value().get<bool>(), meta2, "rungraph");
        }
        continue;
      }

      std::string ec;
      std::string em;
      // Best-effort apply: ignore invalid values rather than rejecting the deploy.
      (void)on_set_state(cfg_.service_id, field, it.value(), meta2, ec, em);
    }

    // Ensure the KV bucket has a full snapshot quickly (reduces monitor "miss" spam).
    publish_static_state();
    publish_dynamic_state();
  } catch (...) {
    // Deploy should not fail because of a local parse issue.
  }

  return true;
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
  else if (call == "stop")
    ok = cmd_stop(err);
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
    (void)open_media_internal(url_to_open, true, err);
  }
}

void ImPlayerService::playlist_play_index(int index) {
  std::string url;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (index < 0 || index >= static_cast<int>(playlist_.size()))
      return;
    playlist_index_ = index;
    url = playlist_[static_cast<std::size_t>(playlist_index_)];
  }
  std::string err;
  (void)open_media_internal(url, true, err);
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
  bool single = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (playlist_.empty())
      return;
    if (playlist_index_ < 0)
      playlist_index_ = 0;
    const int next = playlist_index_ + 1;
    loop = loop_;
    single = (playlist_.size() == 1);
    if (next >= static_cast<int>(playlist_.size())) {
      if (!loop)
        return;
      if (single) {
        url = playlist_[0];
      } else {
        playlist_index_ = 0;
        url = playlist_[0];
      }
    } else {
      playlist_index_ = next;
      url = playlist_[static_cast<std::size_t>(playlist_index_)];
    }
  }
  std::string err;
  if (single) {
    (void)open_media_internal(url, true, err);
    return;
  }
  (void)open_media_internal(url, true, err);
}

void ImPlayerService::playlist_prev() {
  std::string url;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (playlist_.empty())
      return;
    if (playlist_index_ <= 0)
      return;
    playlist_index_ -= 1;
    url = playlist_[static_cast<std::size_t>(playlist_index_)];
  }
  std::string err;
  (void)open_media_internal(url, true, err);
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
        // fallthrough and call mpv loadfile again
      } else {
        if (eof_reached_.load(std::memory_order_acquire)) {
          eof_reached_.store(false, std::memory_order_release);
          media_finished_.store(false, std::memory_order_release);
          player_->seek(0.0);
        }
        if (active_.load(std::memory_order_acquire)) {
          (void)player_->play();
        } else {
          player_->pause();
        }
        return true;
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!apply_auth_options_locked(err)) {
      last_error_ = err;
      return false;
    }
  }

  if (!player_->openMedia(u)) {
    err = "mpv loadfile failed";
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

    vr_auto_video_id_ = video_id_;
    vr_manual_override_ = false;
    const auto detected_from_name = detect_projection_from_name(u);
    vr_auto_detect_mode_ = detected_from_name;
    vr_auto_detect_valid_ = true;
    vr_auto_pending_ratio_ = true;
    vr_mode_ = detected_from_name;
    vr_sbs_eye_ = 0;
    vr_yaw_deg_ = 0.0f;
    vr_pitch_deg_ = 0.0f;
    vr_fov_deg_ = 90.0f;
    vr_dragging_ = false;
  }
  if (active_.load(std::memory_order_acquire)) {
    (void)player_->play();
  } else {
    player_->pause();
  }

  json media_evt;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    media_evt = json{{"videoId", video_id_}, {"url", media_url_}};
  }
  if (bus_) {
    (void)f8::cppsdk::publish_data(bus_->nats(), cfg_.service_id, cfg_.service_id, "media", media_evt);
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
  return open_media_internal(url, false, err);
}

bool ImPlayerService::cmd_play(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  if (stopped_.load(std::memory_order_acquire)) {
    std::string url;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      url = media_url_;
    }
    if (url.empty()) {
      err = "no media loaded";
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (!apply_auth_options_locked(err)) {
        last_error_ = err;
        return false;
      }
    }
    if (!player_->openMedia(url)) {
      err = "mpv loadfile failed";
      return false;
    }
    stopped_.store(false, std::memory_order_release);
    eof_reached_.store(false, std::memory_order_release);
    player_->seek(0.0);
  }
  if (eof_reached_.load(std::memory_order_acquire)) {
    eof_reached_.store(false, std::memory_order_release);
    media_finished_.store(false, std::memory_order_release);
    player_->seek(0.0);
  }
  return player_->play();
}

bool ImPlayerService::cmd_pause(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  player_->pause();
  return true;
}

bool ImPlayerService::cmd_stop(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  player_->stop();
  player_->resetPlaybackState();
  eof_reached_.store(false, std::memory_order_release);
  media_finished_.store(false, std::memory_order_release);
  stopped_.store(true, std::memory_order_release);
  bool cleared_now = false;
  if (window_ && player_) {
    std::unique_lock<std::mutex> render_lock(render_mu_, std::try_to_lock);
    if (render_lock.owns_lock()) {
      (void)window_->makeCurrent();
      player_->resetVideoOutput();
      cleared_now = true;
    }
  }
  if (!cleared_now) {
    clear_video_requested_.store(true, std::memory_order_release);
  }
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

bool ImPlayerService::cmd_seek(const nlohmann::json& args, std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  double pos = 0.0;
  bool ok = false;
  if (args.is_object() && args.contains("position")) {
    if (args["position"].is_number()) {
      pos = args["position"].get<double>();
      ok = true;
    } else if (args["position"].is_string()) {
      try {
        pos = std::stod(args["position"].get<std::string>());
        ok = true;
      } catch (...) {}
    }
  }
  if (!ok) {
    err = "missing position";
    return false;
  }
  if (eof_reached_.load(std::memory_order_acquire) || stopped_.load(std::memory_order_acquire)) {
    std::string url;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      url = media_url_;
    }
    if (url.empty()) {
      err = "no media loaded";
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (!apply_auth_options_locked(err)) {
        last_error_ = err;
        return false;
      }
    }
    if (!player_->openMedia(url)) {
      err = "mpv loadfile failed";
      std::lock_guard<std::mutex> lock(state_mu_);
      last_error_ = err;
      return false;
    }
    eof_reached_.store(false, std::memory_order_release);
    stopped_.store(false, std::memory_order_release);
  }
  player_->seek(pos);
  if (active_.load(std::memory_order_acquire)) {
    (void)player_->play();
  } else {
    player_->pause();
  }
  return true;
}

bool ImPlayerService::cmd_set_volume(const nlohmann::json& args, std::string& err) {
  double vol = 1.0;
  bool ok = false;
  if (args.is_object() && args.contains("volume")) {
    if (args["volume"].is_number()) {
      vol = args["volume"].get<double>();
      ok = true;
    } else if (args["volume"].is_string()) {
      try {
        vol = std::stod(args["volume"].get<std::string>());
        ok = true;
      } catch (...) {}
    }
  }
  if (!ok) {
    err = "missing volume";
    return false;
  }
  vol = std::clamp(vol, 0.0, 1.0);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    volume_ = vol;
  }
  if (player_)
    player_->setVolume(vol);
  return true;
}

void ImPlayerService::mark_vr_manual_override() {
  vr_manual_override_ = true;
  vr_auto_pending_ratio_ = false;
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
    const std::string from_browser = browser_cookie_option_value(auth_browser_, auth_browser_profile_);
    if (!player_->setYtdlRawOptions("cookies-from-browser=" + from_browser)) {
      err = "failed to set ytdl cookies-from-browser";
      return false;
    }
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
  if (!player_->setYtdlRawOptions("")) {
    err = "failed to clear ytdl-raw-options";
    return false;
  }
  if (!player_->setCookiesFile(file)) {
    err = "failed to set cookies-file";
    return false;
  }
  return true;
}

void ImPlayerService::publish_static_state() {
  if (!shm_)
    return;
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
    want("videoShmName", shm_->regionName());
    want("videoShmEvent", shm_->frameEventName());
    want("loop", loop_);
    want("videoShmMaxWidth", cfg_.video_shm_max_width);
    want("videoShmMaxHeight", cfg_.video_shm_max_height);
    want("videoShmMaxFps", cfg_.video_shm_max_fps);
    want("authMode", auth_mode_);
    want("authBrowser", auth_browser_);
  }
  for (const auto& [field, v] : updates) {
    if (bus_) {
      f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
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
    want("lastError", err);

    want("decodedWidth", static_cast<std::int64_t>(decoded_w));
    want("decodedHeight", static_cast<std::int64_t>(decoded_h));

    if (shm_) {
      want("videoWidth", shm_->outputWidth());
      want("videoHeight", shm_->outputHeight());
      want("videoPitch", shm_->outputPitch());
    }
  }

  for (const auto& [field, v] : updates) {
    if (bus_) {
      f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
    }
  }
}

json ImPlayerService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.implayer";
  service["label"] = "IM Player";
  service["version"] = "0.0.1";
  service["description"] = "C++ MPV-based player service with shared-memory video output.";
  service["stateFields"] = json::array({
      state_field("loop", schema_boolean(), "rw", "Loop", "Repeat playlist when reaching EOF.", false),
      state_field("mediaUrl", schema_string(), "rw", "Media URL", "URI or file path to open.", true),
      state_field("volume", schema_number(0.0, 1.0), "rw", "Volume", "0.0-1.0", true),
      state_field("playing", schema_boolean(), "ro", "Playing", "Playback state.", false),
      state_field("duration", schema_number(), "ro", "Duration", "Duration (seconds).", true),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message.", false),
      state_field("videoShmName", schema_string(), "ro", "Video SHM", "Shared memory region name.", true),
      state_field("videoShmEvent", schema_string(), "ro", "Video Event", "Optional named event to signal new frames.",
                  false),
      state_field("videoShmMaxWidth", schema_integer(), "rw", "SHM Max Width", "Downsample limit (0 = auto).", false),
      state_field("videoShmMaxHeight", schema_integer(), "rw", "SHM Max Height", "Downsample limit (0 = auto).", false),
      state_field("videoShmMaxFps", schema_number(), "rw", "SHM Max FPS", "Copy rate limit (0 = unlimited).", false),
      state_field("authMode", schema_string_enum({"none", "browser", "cookiesFile"}), "rw", "Auth Mode",
                  "Cookie auth mode: none|browser|cookiesFile (default: none).", false),
      state_field("authBrowser", schema_string_enum({"chrome", "chromium", "edge", "firefox", "safari"}), "rw",
                  "Auth Browser",
                  "Browser name for authMode=browser: chrome|chromium|edge|firefox|safari.", false),
      state_field("authBrowserProfile", schema_string(), "rw", "Auth Browser Profile",
                  "Optional browser profile for authMode=browser. Sensitive: runtime-only; not persisted.", false),
      state_field("authCookiesFile", schema_string(), "rw", "Auth Cookies File",
                  "cookies.txt path for authMode=cookiesFile. Sensitive: runtime-only; not persisted.", false),
      state_field("decodedWidth", schema_integer(), "ro", "Decoded Width",
                  "Decoded/source video width (on-screen uses this).", false),
      state_field("decodedHeight", schema_integer(), "ro", "Decoded Height",
                  "Decoded/source video height (on-screen uses this).", false),
      state_field("videoWidth", schema_integer(), "ro", "Width", "Width of the video frame.", false),
      state_field("videoHeight", schema_integer(), "ro", "Height", "Height of the video frame.", false),
      state_field("videoPitch", schema_integer(), "ro", "Pitch", "Pitch of the video frame.", false),
  });

  service["dataOutPorts"] = json::array({
      json{{"name", "media"},
           {"valueSchema", schema_object(json{{"videoId", schema_string()}, {"url", schema_string()}},
                                         json::array({"videoId", "url"}))},
           {"description", "Emitted when a new media is opened (videoId + url)."},
           {"required", false},
           {"showOnNode", false}},
      json{{"name", "playback"},
           {"valueSchema", schema_object(json{{"videoId", schema_string()},
                                              {"position", schema_number()},
                                              {"duration", schema_number()},
                                              {"playing", schema_boolean()}},
                                         json::array({"videoId", "position"}))},
           {"description", "Playback telemetry stream (position/duration/playing)."},
           {"required", false},
           {"showOnNode", false}},
      json{{"name", "frameId"},
           {"valueSchema", schema_integer()},
           {"description", "Monotonic frame counter for new shm frames."},
           {"required", false},
           {"showOnNode", false}},
  });
  service["commands"] = json::array({
      json{{"name", "open"},
           {"description", "Open a media URL"},
           {"showOnNode", true},
           {"params", json::array({json{{"name", "url"}, {"valueSchema", schema_string()}, {"required", true}}})}},
      json{{"name", "play"}, {"description", "Start playback"}, {"showOnNode", true}},
      json{{"name", "pause"}, {"description", "Pause playback"}, {"showOnNode", true}},
      json{{"name", "stop"}, {"description", "Stop playback"}, {"showOnNode", true}},
      json{{"name", "seek"},
           {"description", "Seek"},
           {"params", json::array({json{{"name", "position"}, {"valueSchema", schema_number()}, {"required", true}}})}},
      json{{"name", "setVolume"},
           {"description", "Set volume"},
           {"params", json::array({json{{"name", "volume"}, {"valueSchema", schema_number()}, {"required", true}}})}},
  });

  json out;
  out["service"] = service;
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::implayer
