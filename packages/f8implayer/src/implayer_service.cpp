#include "implayer_service.h"

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"
#include "mpv_player.h"
#include "implayer_gui.h"
#include "sdl_video_window.h"
#include "video_shared_memory_sink.h"

namespace f8::implayer {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_number() { return json{{"type", "number"}}; }
json schema_integer() { return json{{"type", "integer"}}; }
json schema_boolean() { return json{{"type", "boolean"}}; }

json state_field(std::string name, const json& value_schema, std::string access, std::string label = {},
                 std::string description = {}, bool show_on_node = false) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  if (!label.empty()) sf["label"] = std::move(label);
  if (!description.empty()) sf["description"] = std::move(description);
  if (show_on_node) sf["showOnNode"] = true;
  return sf;
}

std::string default_video_shm_name(const std::string& service_id) { return "shm." + service_id + ".video"; }

}  // namespace

ImPlayerService::ImPlayerService(Config cfg) : cfg_(std::move(cfg)) {}

ImPlayerService::~ImPlayerService() { stop(); }

bool ImPlayerService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  try {
    cfg_.service_id = f8::cppsdk::ensure_token(cfg_.service_id, "service_id");
  } catch (const std::exception& e) {
    spdlog::error("invalid --service-id: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("invalid --service-id");
    return false;
  }

  if (!nats_.connect(cfg_.nats_url)) return false;

  f8::cppsdk::KvConfig kvc;
  kvc.bucket = f8::cppsdk::kv_bucket_for_service(cfg_.service_id);
  kvc.history = 1;
  kvc.memory_storage = true;
  if (!kv_.open_or_create(nats_.jetstream(), kvc)) return false;

  ctrl_ = std::make_unique<f8::cppsdk::ServiceControlPlaneServer>(
      f8::cppsdk::ServiceControlPlaneServer::Config{cfg_.service_id, cfg_.nats_url}, &nats_, &kv_, this);
  if (!ctrl_->start()) {
    spdlog::error("failed to start control plane");
    return false;
  }

  shm_ = std::make_shared<VideoSharedMemorySink>();
  const auto shm_name = default_video_shm_name(cfg_.service_id);
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
        [this]() { playing_.store(false, std::memory_order_relaxed); });
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
  if (!active_.load(std::memory_order_acquire)) {
    player_->pause();
  }
  if (!cfg_.initial_media_url.empty()) {
    std::string err;
    if (!cmd_open(json{{"url", cfg_.initial_media_url}}, err)) {
      spdlog::error("initial --media failed: {}", err);
    }
  }

  publish_static_state();
  f8::cppsdk::kv_set_ready(kv_, true);

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("implayer started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void ImPlayerService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  stop_requested_.store(true, std::memory_order_release);

  try {
    if (ctrl_) ctrl_->stop();
  } catch (...) {
  }
  ctrl_.reset();

  if (window_) window_->makeCurrent();
  if (player_) player_->shutdownGl();
  if (gui_) gui_->stop();
  gui_.reset();
  player_.reset();
  window_.reset();
  shm_.reset();
  kv_.stop_watch();
  kv_.close();
  nats_.close();
}

void ImPlayerService::tick() {
  if (!running_.load(std::memory_order_acquire)) return;

  if (window_) {
    bool saw_input = false;
    auto on_ev = [this, &saw_input](const SDL_Event& ev) {
      SDL_Event copy = ev;
      if (gui_) gui_->processEvent(&copy);
      if (ev.type == SDL_EVENT_KEY_DOWN) {
        saw_input = true;
        if (!player_) return;
        const SDL_Keycode key = ev.key.key;
        if (key == SDLK_SPACE) {
          // Toggle pause by reading pause property is expensive; just call play() then pause() based on last playing_.
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
      } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN || ev.type == SDL_EVENT_MOUSE_WHEEL ||
                 ev.type == SDL_EVENT_MOUSE_MOTION) {
        saw_input = true;
      }
    };

    if (!window_->pumpEvents(on_ev)) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
    (void)saw_input;
  }
  if (player_ && window_) {
    const bool updated = player_->renderVideoFrame();
    if (updated || window_->needsRedraw() || (gui_ && gui_->wantsRepaint())) {
      ImPlayerGui::Callbacks cb;
      cb.open = [this](const std::string& url) {
        std::string err;
        (void)cmd_open(json{{"url", url}}, err);
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

      std::string err;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        err = last_error_;
      }

      window_->present(*player_, [this, &cb, &err]() {
        if (gui_ && player_) {
          gui_->renderOverlay(*player_, cb, err);
          gui_->clearRepaintFlag();
        }
      });
    }
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (now - last_state_pub_ms_ >= 200) {
    publish_dynamic_state();
    last_state_pub_ms_ = now;
  }
}

void ImPlayerService::set_active_local(bool active, const nlohmann::json& meta) {
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
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "active", active, "cmd", meta);
}

void ImPlayerService::on_activate(const nlohmann::json& meta) { set_active_local(true, meta); }
void ImPlayerService::on_deactivate(const nlohmann::json& meta) { set_active_local(false, meta); }
void ImPlayerService::on_set_active(bool active, const nlohmann::json& meta) { set_active_local(active, meta); }

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
  } else if (f == "active") {
    if (!value.is_boolean()) {
      err = "active must be boolean";
      ok = false;
    } else {
      set_active_local(value.get<bool>(), meta);
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

  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, f, value, "endpoint", meta);
  return true;
}

bool ImPlayerService::on_set_rungraph(const nlohmann::json&, const nlohmann::json&, std::string& error_code,
                                     std::string& error_message) {
  // For now, accept any rungraph snapshot; operator nodes are not implemented in C++ yet.
  error_code.clear();
  error_message.clear();
  return true;
}

bool ImPlayerService::on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                                nlohmann::json& result, std::string& error_code, std::string& error_message) {
  std::string err;
  bool ok = false;

  if (call == "open") ok = cmd_open(args, err);
  else if (call == "play") ok = cmd_play(err);
  else if (call == "pause") ok = cmd_pause(err);
  else if (call == "stop") ok = cmd_stop(err);
  else if (call == "seek") ok = cmd_seek(args, err);
  else if (call == "setVolume") ok = cmd_set_volume(args, err);
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

bool ImPlayerService::cmd_open(const nlohmann::json& args, std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
  }
  std::string url;
  if (args.is_object()) {
    if (args.contains("url") && args["url"].is_string()) url = args["url"].get<std::string>();
    if (url.empty() && args.contains("mediaUrl") && args["mediaUrl"].is_string()) url = args["mediaUrl"].get<std::string>();
  }
  if (url.empty()) {
    err = "missing url";
    return false;
  }
  if (!player_->openMedia(url)) {
    err = "mpv loadfile failed";
    std::lock_guard<std::mutex> lock(state_mu_);
    last_error_ = err;
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    media_url_ = url;
    last_error_.clear();
  }
  if (active_.load(std::memory_order_acquire)) {
    player_->play();
  } else {
    player_->pause();
  }
  return true;
}

bool ImPlayerService::cmd_play(std::string& err) {
  if (!player_) {
    err = "player not initialized";
    return false;
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
      } catch (...) {
      }
    }
  }
  if (!ok) {
    err = "missing position";
    return false;
  }
  player_->seek(pos);
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
      } catch (...) {
      }
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
  if (player_) player_->setVolume(vol);
  return true;
}

void ImPlayerService::publish_static_state() {
  if (!shm_) return;
  const json meta = json{{"via", "startup"}};

  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "serviceClass", cfg_.service_class, "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoShmName", shm_->regionName(), "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoShmEvent", shm_->frameEventName(), "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "active", active_.load(), "runtime", meta);
}

void ImPlayerService::publish_dynamic_state() {
  const json meta = json{{"via", "periodic"}};

  double vol = 1.0;
  std::string url;
  std::string err;
  const bool playing = playing_.load(std::memory_order_relaxed);
  const double pos = position_seconds_.load(std::memory_order_relaxed);
  const double dur = duration_seconds_.load(std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    vol = volume_;
    url = media_url_;
    err = last_error_;
  }

  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "playing", playing, "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "position", pos, "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "duration", dur, "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "volume", vol, "runtime", meta);
  f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "mediaUrl", url, "runtime", meta);

  if (shm_) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoWidth", shm_->outputWidth(), "runtime", meta);
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoHeight", shm_->outputHeight(), "runtime", meta);
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoPitch", shm_->outputPitch(), "runtime", meta);
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "videoFrameId", shm_->frameId(), "runtime", meta);
  }
  if (!err.empty()) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "lastError", err, "runtime", meta);
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
      state_field("active", schema_boolean(), "rw", "Active", "Pause playback when false.", true),
      state_field("mediaUrl", schema_string(), "rw", "Media URL", "URI or file path to open.", true),
      state_field("volume", schema_number(), "rw", "Volume", "0.0-1.0"),
      state_field("position", schema_number(), "rw", "Position", "Seek position (seconds)."),
      state_field("playing", schema_boolean(), "ro", "Playing", "Playback state.", true),
      state_field("duration", schema_number(), "ro", "Duration", "Duration (seconds)."),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message."),
      state_field("videoShmName", schema_string(), "ro", "Video SHM", "Shared memory region name."),
      state_field("videoShmEvent", schema_string(), "ro", "Video Event", "Optional named event to signal new frames."),
      state_field("videoWidth", schema_integer(), "ro", "Width"),
      state_field("videoHeight", schema_integer(), "ro", "Height"),
      state_field("videoPitch", schema_integer(), "ro", "Pitch"),
      state_field("videoFrameId", schema_integer(), "ro", "FrameId"),
  });
  service["commands"] = json::array({
      json{{"name", "open"}, {"description", "Open a media URL"}, {"params", json::array({json{{"name", "url"}, {"valueSchema", schema_string()}, {"required", true}}})}},
      json{{"name", "play"}, {"description", "Start playback"}},
      json{{"name", "pause"}, {"description", "Pause playback"}},
      json{{"name", "stop"}, {"description", "Stop playback"}},
      json{{"name", "seek"}, {"description", "Seek"}, {"params", json::array({json{{"name", "position"}, {"valueSchema", schema_number()}, {"required", true}}})}},
      json{{"name", "setVolume"}, {"description", "Set volume"}, {"params", json::array({json{{"name", "volume"}, {"valueSchema", schema_number()}, {"required", true}}})}},
  });

  json out;
  out["service"] = service;
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::implayer
