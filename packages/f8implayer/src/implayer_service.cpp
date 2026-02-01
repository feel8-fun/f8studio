#include "implayer_service.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <string>
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

  // Start the service bus only after the GUI/player are ready, so rungraph/state
  // deployments won't race against initialization.
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(f8::cppsdk::ServiceBus::Config{cfg_.service_id, cfg_.nats_url, true});
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

      const bool updated = player_->renderVideoFrame();
      if (updated || window_->needsRedraw() || (gui_ && gui_->wantsRepaint())) {
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
        cb.playlist_select = [this](int index) {
          playlist_play_index(index);
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
        const bool playing = playing_.load(std::memory_order_relaxed);
        window_->present(
            *player_,
            [this, &cb, &err, &playlist_snapshot, playlist_index_snapshot, playing, loop_snapshot]() {
              if (gui_ && player_) {
                gui_->renderOverlay(*player_, cb, err, playlist_snapshot, playlist_index_snapshot, playing, loop_snapshot);
                gui_->clearRepaintFlag();
              }
            },
            view);
      }
    }
  }

  const std::int64_t now = f8::cppsdk::now_ms();
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
    if (ev.button.button == SDL_BUTTON_MIDDLE) {
      view_panning_ = false;
    }
    return;
  }

  if (ev.type == SDL_EVENT_MOUSE_MOTION) {
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
    const SDL_Keycode key = ev.key.key;
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
  } else if (f == "active") {
    if (!value.is_boolean()) {
      err = "active must be boolean";
      ok = false;
    } else {
      if (!bus_) {
        err = "service bus not ready";
        ok = false;
      } else {
        bus_->set_active_local(value.get<bool>(), meta, "endpoint");
        ok = true;
      }
    }
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
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = published_state_.find(f);
    if (it == published_state_.end() || it->second != write_value) {
      published_state_[f] = write_value;
      if (bus_) {
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
          field != "videoShmMaxHeight" && field != "videoShmMaxFps") {
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
    const bool empty_before = playlist_.empty();
    for (const auto& s : items) {
      if (!s.empty())
        playlist_.push_back(s);
    }
    if (playlist_.empty())
      return;
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
    if (!player_) {
      return;
    }
    if (!player_->openMedia(url)) {
      err = "mpv loadfile failed";
      std::lock_guard<std::mutex> lock(state_mu_);
      last_error_ = err;
      return;
    }
    eof_reached_.store(false, std::memory_order_release);
    player_->seek(0.0);
    if (active_.load(std::memory_order_acquire)) {
      (void)player_->play();
    } else {
      player_->pause();
    }
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
  const std::string u = trim_copy(url);
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
      if (active_.load(std::memory_order_acquire)) {
        (void)player_->play();
      } else {
        player_->pause();
      }
      return true;
    }
  }

  if (!player_->openMedia(u)) {
    err = "mpv loadfile failed";
    std::lock_guard<std::mutex> lock(state_mu_);
    last_error_ = err;
    return false;
  }
  eof_reached_.store(false, std::memory_order_release);
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
  return open_media_internal(url, false, err);
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
  eof_reached_.store(false, std::memory_order_release);
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
  if (eof_reached_.load(std::memory_order_acquire)) {
    std::string url;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      url = media_url_;
    }
    if (url.empty()) {
      err = "no media loaded";
      return false;
    }
    if (!player_->openMedia(url)) {
      err = "mpv loadfile failed";
      std::lock_guard<std::mutex> lock(state_mu_);
      last_error_ = err;
      return false;
    }
    eof_reached_.store(false, std::memory_order_release);
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
    want("active", active_.load());
    want("loop", loop_);
    want("videoShmMaxWidth", cfg_.video_shm_max_width);
    want("videoShmMaxHeight", cfg_.video_shm_max_height);
    want("videoShmMaxFps", cfg_.video_shm_max_fps);
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
      state_field("active", schema_boolean(), "rw", "Active", "Pause playback when false.", true),
      state_field("loop", schema_boolean(), "rw", "Loop", "Repeat playlist when reaching EOF.", false),
      state_field("mediaUrl", schema_string(), "rw", "Media URL", "URI or file path to open.", true),
      state_field("volume", schema_number(0.0, 1.0), "rw", "Volume", "0.0-1.0", false, "slider"),
      state_field("playing", schema_boolean(), "ro", "Playing", "Playback state.", true),
      state_field("duration", schema_number(), "ro", "Duration", "Duration (seconds)."),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message."),
      state_field("videoShmName", schema_string(), "ro", "Video SHM", "Shared memory region name.", true),
      state_field("videoShmEvent", schema_string(), "ro", "Video Event", "Optional named event to signal new frames."),
      state_field("videoShmMaxWidth", schema_integer(), "rw", "SHM Max Width", "Downsample limit (0 = auto)."),
      state_field("videoShmMaxHeight", schema_integer(), "rw", "SHM Max Height", "Downsample limit (0 = auto)."),
      state_field("videoShmMaxFps", schema_number(), "rw", "SHM Max FPS", "Copy rate limit (0 = unlimited)."),
      state_field("decodedWidth", schema_integer(), "ro", "Decoded Width", "Decoded/source video width (on-screen uses this).", true),
      state_field("decodedHeight", schema_integer(), "ro", "Decoded Height", "Decoded/source video height (on-screen uses this).", true),
      state_field("videoWidth", schema_integer(), "ro", "Width", "Width of the video frame.", true),
      state_field("videoHeight", schema_integer(), "ro", "Height", "Height of the video frame.", true),
      state_field("videoPitch", schema_integer(), "ro", "Pitch", "Pitch of the video frame."),
  });

  service["dataOutPorts"] = json::array({
      json{{"name", "media"},
           {"valueSchema", schema_object(json{{"videoId", schema_string()}, {"url", schema_string()}},
                                         json::array({"videoId", "url"}))},
           {"description", "Emitted when a new media is opened (videoId + url)."},
           {"required", false}},
      json{{"name", "playback"},
           {"valueSchema", schema_object(json{{"videoId", schema_string()},
                                              {"position", schema_number()},
                                              {"duration", schema_number()},
                                              {"playing", schema_boolean()}},
                                         json::array({"videoId", "position"}))},
           {"description", "Playback telemetry stream (position/duration/playing)."},
           {"required", false}},
      json{{"name", "frameId"},
           {"valueSchema", schema_integer()},
           {"description", "Monotonic frame counter for new shm frames."},
           {"required", false}},
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
