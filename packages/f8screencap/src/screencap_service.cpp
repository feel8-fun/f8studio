#include "screencap_service.h"

#include <algorithm>
#include <sstream>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/shm/video.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/video_shared_memory_sink.h"

#if defined(_WIN32)
#include "win32_capture_sources.h"
#include "win32_picker.h"
#include "win32_wgc_capture.h"
#else
#include "linux_x11_capture.h"
#include "x11_capture_sources.h"
#include "x11_picker.h"
#endif

namespace f8::screencap {

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

bool is_mode_valid(const std::string& m) {
  return m == "display" || m == "window" || m == "region";
}

json rect_schema() {
  return schema_object(
      json{{"x", schema_integer()}, {"y", schema_integer()}, {"w", schema_integer()}, {"h", schema_integer()}},
      json::array({"x", "y", "w", "h"}));
}

json size_schema() {
  return schema_object(json{{"w", schema_integer()}, {"h", schema_integer()}}, json::array({"w", "h"}));
}

bool coerce_rect_csv(const json& value, std::string& out_csv, std::string& err) {
  err.clear();
  if (value.is_string()) {
    out_csv = value.get<std::string>();
    return true;
  }
  if (!value.is_object()) {
    err = "expected rect object or csv string";
    return false;
  }
  const int x = value.value("x", 0);
  const int y = value.value("y", 0);
  const int w = value.value("w", 0);
  const int h = value.value("h", 0);
  if (w <= 0 || h <= 0) {
    err = "w/h must be > 0";
    return false;
  }
  out_csv = std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h);
  return true;
}

bool coerce_size_csv(const json& value, std::string& out_csv, std::string& err) {
  err.clear();
  if (value.is_null()) {
    out_csv.clear();
    return true;
  }
  if (value.is_string()) {
    out_csv = value.get<std::string>();
    return true;
  }
  if (!value.is_object()) {
    err = "expected size object or csv string";
    return false;
  }
  const int w = value.value("w", 0);
  const int h = value.value("h", 0);
  if (w < 0 || h < 0) {
    err = "w/h must be >= 0";
    return false;
  }
  out_csv = std::to_string(w) + "," + std::to_string(h);
  return true;
}

json rect_object_from_csv_best_effort(const std::string& csv) {
  int x = 0, y = 0, w = 0, h = 0;
  try {
    std::string s = csv;
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream ss(s);
    ss >> x >> y >> w >> h;
  } catch (...) {
    x = y = w = h = 0;
  }
  return json{{"x", x}, {"y", y}, {"w", w}, {"h", h}};
}

json size_object_from_csv_best_effort(const std::string& csv) {
  int w = 0, h = 0;
  try {
    std::string s = csv;
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream ss(s);
    ss >> w >> h;
  } catch (...) {
    w = h = 0;
  }
  return json{{"w", w}, {"h", h}};
}

}  // namespace

ScreenCapService::ScreenCapService(Config cfg) : cfg_(std::move(cfg)) {}

ScreenCapService::~ScreenCapService() {
  stop();
}

void ScreenCapService::on_lifecycle(bool active, const json& meta) {
  set_active_local(active, meta);
}

void ScreenCapService::on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                                std::int64_t ts_ms, const nlohmann::json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id)
    return;
  std::string ec;
  std::string em;
  nlohmann::json result;
  (void)on_set_state(node_id, field, value, meta, ec, em);
}

bool ScreenCapService::start() {
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

  shm_ = std::make_shared<f8::cppsdk::VideoSharedMemorySink>();
  const auto shm_name = f8::cppsdk::shm::video_shm_name(cfg_.service_id);
  if (!shm_->initialize(shm_name, cfg_.video_shm_bytes, cfg_.video_shm_slots)) {
    spdlog::error("failed to initialize video shm sink name={} bytes={} slots={}", shm_name, cfg_.video_shm_bytes,
                  cfg_.video_shm_slots);
    return false;
  }

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
    shm_.reset();
    return false;
  }

#if defined(_WIN32)
  capture_ = std::make_unique<Win32WgcCapture>(cfg_.service_id, shm_);
#else
  capture_ = std::make_unique<LinuxX11Capture>(cfg_.service_id, shm_);
#endif
  capture_->configure(cfg_.mode, cfg_.fps, cfg_.display_id, cfg_.window_id, cfg_.region_csv, cfg_.scale_csv);
  capture_->set_on_running([this](bool r) { capture_running_.store(r, std::memory_order_release); });
  capture_->set_on_frame([this](std::uint64_t frame_id, std::int64_t ts_ms) {
    frame_id_.store(frame_id, std::memory_order_release);
    last_frame_ts_ms_.store(ts_ms, std::memory_order_release);
  });
  capture_->set_on_error([this](std::string err) {
    std::lock_guard<std::mutex> lock(state_mu_);
    last_error_ = std::move(err);
  });
  // Default: do not start capture until we receive a "deployment/config signal"
  // (e.g. set_rungraph) or external state writes for capture fields.
  capture_->set_active(false);

  // Seed persistent fields.
  publish_static_state();
  publish_dynamic_state();

  running_.store(true, std::memory_order_release);
  spdlog::info("screencap started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void ScreenCapService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;

  if (capture_) {
    capture_.reset();
  }
  if (bus_) {
    try {
      bus_->stop();
    } catch (...) {}
  }
  bus_.reset();
  shm_.reset();
}

void ScreenCapService::tick() {
  if (!running())
    return;

  if (bus_) {
    (void)bus_->drain_main_thread();
  }

  if (bus_ && bus_->terminate_requested()) {
    stop_requested_.store(true, std::memory_order_release);
    return;
  }

  // Apply pending picker result on the main thread (so KV writes are serialized).
  {
    std::optional<json> patch;
    {
      std::lock_guard<std::mutex> lock(picker_mu_);
      if (picker_pending_patch_) {
        patch = std::move(picker_pending_patch_);
        picker_pending_patch_.reset();
      }
    }
    if (patch && patch->is_object()) {
      json meta = json::object({{"via", "picker"}});
      std::string ec, em;
      for (auto it = patch->begin(); it != patch->end(); ++it) {
        const std::string field = it.key();
        const json value = it.value();
        if (field == "window") {
          std::lock_guard<std::mutex> lock(state_mu_);
          published_state_["window"] = value;
          if (bus_) {
            f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, "window", value, "picker",
                                          meta);
          }
          continue;
        }
        (void)on_set_state(cfg_.service_id, field, value, meta, ec, em);
      }
    }
  }

  // Only run capture after we're "armed" by rungraph deploy or external configuration.
  const bool should_capture = active_.load(std::memory_order_acquire) && armed_.load(std::memory_order_acquire);
  if (capture_)
    capture_->set_active(should_capture);

  if (should_capture && capture_restart_.exchange(false, std::memory_order_acq_rel)) {
    if (capture_)
      capture_->restart();
  } else {
    (void)capture_restart_.exchange(false, std::memory_order_acq_rel);
  }

  if (capture_)
    capture_->tick();

  const auto now = f8::cppsdk::now_ms();
  if (last_state_pub_ms_ == 0 || now - last_state_pub_ms_ >= 250) {
    publish_dynamic_state();
    last_state_pub_ms_ = now;
  }
}

void ScreenCapService::set_active_local(bool active, const json&) {
  active_.store(active, std::memory_order_release);
  if (capture_)
    capture_->set_active(active && armed_.load(std::memory_order_acquire));
}

bool ScreenCapService::on_set_state(const std::string& node_id, const std::string& field, const json& value,
                                    const json& meta, std::string& error_code, std::string& error_message) {
  error_code.clear();
  error_message.clear();

  if (node_id != cfg_.service_id) {
    error_code = "INVALID_ARGS";
    error_message = "nodeId must equal serviceId for service node state";
    return false;
  }

  const std::string f = field;
  std::string err;
  bool ok = false;

  if (f == "mode") {
    if (!value.is_string()) {
      err = "mode must be string";
      ok = false;
    } else {
      const std::string m = value.get<std::string>();
      if (!is_mode_valid(m)) {
        err = "mode must be one of: display|window|region";
        ok = false;
      } else {
        cfg_.mode = m;
        ok = true;
      }
    }
  } else if (f == "fps") {
    if (!value.is_number()) {
      err = "fps must be number";
      ok = false;
    } else {
      const double fps = value.get<double>();
      if (!(fps > 0.0) || fps > 240.0) {
        err = "fps must be in (0, 240]";
        ok = false;
      } else {
        cfg_.fps = fps;
        ok = true;
      }
    }
  } else if (f == "displayId") {
    if (!value.is_number_integer() && !value.is_number()) {
      err = "displayId must be integer";
      ok = false;
    } else {
      cfg_.display_id = static_cast<int>(value.get<double>());
      ok = true;
    }
  } else if (f == "windowId") {
    if (!value.is_string()) {
      err = "windowId must be string";
      ok = false;
    } else {
      cfg_.window_id = value.get<std::string>();
      ok = true;
    }
  } else if (f == "region") {
    std::string csv;
    if (!coerce_rect_csv(value, csv, err)) {
      ok = false;
    } else {
      cfg_.region_csv = std::move(csv);
      ok = true;
    }
  } else if (f == "scale") {
    std::string csv;
    if (!coerce_size_csv(value, csv, err)) {
      ok = false;
    } else {
      cfg_.scale_csv = std::move(csv);
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

  // Any successful capture-config state write arms capture (even if currently inactive).
  armed_.store(true, std::memory_order_release);

  if (capture_) {
    capture_->configure(cfg_.mode, cfg_.fps, cfg_.display_id, cfg_.window_id, cfg_.region_csv, cfg_.scale_csv);
  }
  request_capture_restart();

  // Persist (best-effort) and dedupe.
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    json write_value = value;
    if (f == "displayId")
      write_value = cfg_.display_id;
    if (f == "fps")
      write_value = cfg_.fps;
    if (f == "mode")
      write_value = cfg_.mode;
    if (f == "windowId")
      write_value = cfg_.window_id;

    // For region/scale, keep both the structured and the original csv (for CLI compatibility).
    if (f == "region") {
      write_value = value.is_object() ? value : rect_object_from_csv_best_effort(cfg_.region_csv);
    } else if (f == "scale") {
      if (value.is_object()) {
        write_value = value;
      } else if (!cfg_.scale_csv.empty()) {
        write_value = size_object_from_csv_best_effort(cfg_.scale_csv);
      } else {
        write_value = json{{"w", 0}, {"h", 0}};
      }
    }

    auto it = published_state_.find(f);
    if (it == published_state_.end() || it->second != write_value) {
      published_state_[f] = write_value;
      if (bus_) {
        f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, f, write_value, "endpoint", meta);
      }
    }

#if defined(_WIN32)
    if (f == "windowId") {
      json w = json::object();
      if (!cfg_.window_id.empty()) {
        std::uintptr_t hwnd = 0;
        std::string err2;
        if (win32::try_parse_window_id(cfg_.window_id, hwnd, err2)) {
          win32::RectI rc{};
          std::string title;
          std::uint32_t pid = 0;
          if (win32::try_get_window_rect(hwnd, rc, title, pid, err2)) {
            w = json{{"backend", "win32"},
                     {"id", cfg_.window_id},
                     {"pid", pid},
                     {"title", title},
                     {"rect", json{{"x", rc.x}, {"y", rc.y}, {"w", rc.w}, {"h", rc.h}}}};
          }
        }
      }
      published_state_["window"] = w;
      if (bus_) {
        f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, "window", w, "endpoint", meta);
      }
    }
#else
    if (f == "windowId") {
      json w = json::object();
      if (!cfg_.window_id.empty()) {
        std::uint64_t xid = 0;
        std::string err2;
        if (x11::try_parse_window_id(cfg_.window_id, xid, err2)) {
          x11::RectI rc{};
          std::string title;
          std::uint32_t pid = 0;
          if (x11::try_get_window_rect(xid, rc, title, pid, err2)) {
            w = json{{"backend", "x11"},
                     {"id", cfg_.window_id},
                     {"pid", pid},
                     {"title", title},
                     {"rect", json{{"x", rc.x}, {"y", rc.y}, {"w", rc.w}, {"h", rc.h}}}};
          }
        }
      }
      published_state_["window"] = w;
      if (bus_) {
        f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, "window", w, "endpoint", meta);
      }
    }
#endif
  }
  return true;
}

bool ScreenCapService::on_set_rungraph(const json& graph_obj, const json& meta, std::string& error_code,
                                       std::string& error_message) {
  error_code.clear();
  error_message.clear();

  // Treat any rungraph deploy as the "start signal" for this service.
  armed_.store(true, std::memory_order_release);

  try {
    if (!graph_obj.is_object() || !graph_obj.contains("nodes") || !graph_obj["nodes"].is_array())
      return true;

    const auto nodes = graph_obj["nodes"];
    json service_node;
    for (const auto& n : nodes) {
      if (!n.is_object())
        continue;
      const std::string nid = n.value("nodeId", "");
      if (nid != cfg_.service_id)
        continue;
      service_node = n;
      break;
    }
    if (!service_node.is_object() || !service_node.contains("stateValues") || !service_node["stateValues"].is_object())
      return true;

    json meta2 = meta.is_object() ? meta : json::object();
    meta2["via"] = "rungraph";
    meta2["graphId"] = graph_obj.value("graphId", "");

    const auto& values = service_node["stateValues"];
    for (auto it = values.begin(); it != values.end(); ++it) {
      const std::string field = it.key();
      if (field.empty())
        continue;
      if (field != "active" && field != "mode" && field != "fps" && field != "displayId" && field != "windowId" &&
          field != "region" && field != "scale") {
        continue;
      }
      if (field == "active") {
        if (bus_ && it.value().is_boolean()) {
          bus_->set_active_local(it.value().get<bool>(), meta2, "rungraph");
        }
        continue;
      }
      std::string ec, em;
      (void)on_set_state(cfg_.service_id, field, it.value(), meta2, ec, em);
    }
  } catch (...) {
    return true;
  }
  return true;
}

bool ScreenCapService::on_command(const std::string& call, const json& args, const json&, json& result,
                                  std::string& error_code, std::string& error_message) {
  error_code.clear();
  error_message.clear();
  result = json::object();

  if (call == "listDisplays") {
#if defined(_WIN32)
    const auto mons = win32::enumerate_monitors();
    json arr = json::array();
    for (const auto& m : mons) {
      arr.push_back(json{
          {"displayId", m.id},
          {"name", m.name},
          {"primary", m.primary},
          {"rect", json{{"x", m.rect.x}, {"y", m.rect.y}, {"w", m.rect.w}, {"h", m.rect.h}}},
          {"workRect", json{{"x", m.work_rect.x}, {"y", m.work_rect.y}, {"w", m.work_rect.w}, {"h", m.work_rect.h}}}});
    }
    result["displays"] = std::move(arr);
    return true;
#else
    std::string err;
    const auto displays = x11::enumerate_displays(err);
    if (displays.empty()) {
      error_code = "NOT_SUPPORTED";
      error_message = err.empty() ? "listDisplays not supported" : err;
      return false;
    }
    json arr = json::array();
    for (const auto& d : displays) {
      arr.push_back(json{
          {"displayId", d.id},
          {"name", d.name},
          {"primary", d.primary},
          {"rect", json{{"x", d.rect.x}, {"y", d.rect.y}, {"w", d.rect.w}, {"h", d.rect.h}}},
          {"workRect", json{{"x", d.work_rect.x}, {"y", d.work_rect.y}, {"w", d.work_rect.w}, {"h", d.work_rect.h}}}});
    }
    result["displays"] = std::move(arr);
    return true;
#endif
  }

  if (call == "pickDisplay" || call == "pickWindow" || call == "pickRegion") {
    if (picker_running_.exchange(true, std::memory_order_acq_rel)) {
      error_code = "BUSY";
      error_message = "picker already running";
      return false;
    }

#if !defined(_WIN32)
    (void)args;
    if (call == "pickRegion") {
      x11::X11Picker::pick_region_async([this](x11::PickRegionResult r) {
        picker_running_.store(false, std::memory_order_release);
        if (!r.ok) {
          if (!r.error.empty()) {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = "pickRegion: " + r.error;
          }
          return;
        }
        json patch;
        patch["mode"] = "region";
        patch["region"] = json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}};
        std::lock_guard<std::mutex> lock(picker_mu_);
        picker_pending_patch_ = std::move(patch);
      });
    } else if (call == "pickDisplay") {
      x11::X11Picker::pick_display_async([this](x11::PickDisplayResult r) {
        picker_running_.store(false, std::memory_order_release);
        if (!r.ok) {
          if (!r.error.empty()) {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = "pickDisplay: " + r.error;
          }
          return;
        }
        json patch;
        patch["mode"] = "display";
        patch["displayId"] = r.display_id;
        patch["region"] = json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}};
        std::lock_guard<std::mutex> lock(picker_mu_);
        picker_pending_patch_ = std::move(patch);
      });
    } else if (call == "pickWindow") {
      x11::X11Picker::pick_window_async([this](x11::PickWindowResult r) {
        picker_running_.store(false, std::memory_order_release);
        if (!r.ok) {
          if (!r.error.empty()) {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_ = "pickWindow: " + r.error;
          }
          return;
        }
        json patch;
        patch["mode"] = "window";
        patch["windowId"] = r.window_id;
        patch["region"] = json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}};
        patch["window"] = json{{"backend", "x11"},
                               {"id", r.window_id},
                               {"pid", r.pid},
                               {"title", r.title},
                               {"rect", json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}}}};
        std::lock_guard<std::mutex> lock(picker_mu_);
        picker_pending_patch_ = std::move(patch);
      });
    } else {
      picker_running_.store(false, std::memory_order_release);
      error_code = "NOT_SUPPORTED";
      error_message = "picker not supported on this platform";
      return false;
    }
    result["started"] = true;
    return true;
#else
    (void)args;
    if (call == "pickDisplay") {
      win32::Win32Picker::pick_display_async([this](win32::PickDisplayResult r) {
        picker_running_.store(false, std::memory_order_release);
        if (!r.ok)
          return;
        json patch;
        patch["mode"] = "display";
        patch["displayId"] = r.display_id;
        std::lock_guard<std::mutex> lock(picker_mu_);
        picker_pending_patch_ = std::move(patch);
      });
    } else if (call == "pickWindow") {
      win32::Win32Picker::pick_window_async([this](win32::PickWindowResult r) {
        picker_running_.store(false, std::memory_order_release);
        if (!r.ok)
          return;
        json patch;
        patch["mode"] = "window";
        patch["windowId"] = r.window_id;
        patch["region"] = json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}};
        patch["window"] = json{{"backend", "win32"},
                               {"id", r.window_id},
                               {"pid", r.pid},
                               {"title", r.title},
                               {"rect", json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}}}};
        std::lock_guard<std::mutex> lock(picker_mu_);
        picker_pending_patch_ = std::move(patch);
      });
    } else if (call == "pickRegion") {
      win32::Win32Picker::pick_region_async([this](win32::PickRegionResult r) {
        picker_running_.store(false, std::memory_order_release);
        if (!r.ok)
          return;
        json patch;
        patch["mode"] = "region";
        patch["region"] = json{{"x", r.rect.x}, {"y", r.rect.y}, {"w", r.rect.w}, {"h", r.rect.h}};
        std::lock_guard<std::mutex> lock(picker_mu_);
        picker_pending_patch_ = std::move(patch);
      });
    }
    result["started"] = true;
    return true;
#endif
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void ScreenCapService::publish_static_state() {
  std::lock_guard<std::mutex> lock(state_mu_);

  auto set_if_changed = [&](const char* field, const json& v) {
    auto it = published_state_.find(field);
    if (it != published_state_.end() && it->second == v)
      return;
    published_state_[field] = v;
    if (bus_)
      f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, v, "init", json::object());
  };

  set_if_changed("serviceClass", cfg_.service_class);
  set_if_changed("videoShmName", shm_ ? shm_->regionName() : "");
  set_if_changed("videoShmEvent", shm_ ? shm_->frameEventName() : "");

  set_if_changed("mode", cfg_.mode);
  set_if_changed("fps", cfg_.fps);
  set_if_changed("displayId", cfg_.display_id);
  set_if_changed("windowId", cfg_.window_id);
  set_if_changed("window", json::object());
  {
    json region = rect_object_from_csv_best_effort(cfg_.region_csv);
#if defined(_WIN32)
    // If region isn't set, seed from the active source (best-effort).
    if ((region.value("w", 0) <= 0 || region.value("h", 0) <= 0) && cfg_.mode == "display") {
      win32::RectI r{};
      std::string err;
      if (win32::try_get_monitor_rect(cfg_.display_id, r, err)) {
        region = json{{"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h}};
      }
    } else if ((region.value("w", 0) <= 0 || region.value("h", 0) <= 0) && cfg_.mode == "window") {
      std::uintptr_t hwnd = 0;
      std::string err;
      if (win32::try_parse_window_id(cfg_.window_id, hwnd, err)) {
        win32::RectI r{};
        std::string title;
        std::uint32_t pid = 0;
        if (win32::try_get_window_rect(hwnd, r, title, pid, err)) {
          region = json{{"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h}};
        }
      }
    }
#else
    if ((region.value("w", 0) <= 0 || region.value("h", 0) <= 0) && cfg_.mode == "display") {
      std::string err;
      const auto displays = x11::enumerate_displays(err);
      if (!displays.empty()) {
        const auto* sel = &displays.front();
        for (const auto& d : displays) {
          if (d.id == cfg_.display_id) {
            sel = &d;
            break;
          }
        }
        const auto& d = *sel;
        region = json{{"x", d.rect.x}, {"y", d.rect.y}, {"w", d.rect.w}, {"h", d.rect.h}};
      }
    } else if ((region.value("w", 0) <= 0 || region.value("h", 0) <= 0) && cfg_.mode == "window") {
      std::uint64_t xid = 0;
      std::string err;
      if (x11::try_parse_window_id(cfg_.window_id, xid, err)) {
        x11::RectI r{};
        std::string title;
        std::uint32_t pid = 0;
        if (x11::try_get_window_rect(xid, r, title, pid, err)) {
          region = json{{"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h}};
        }
      }
    }
#endif
    set_if_changed("region", region);
  }
  set_if_changed("scale",
                 cfg_.scale_csv.empty() ? json{{"w", 0}, {"h", 0}} : size_object_from_csv_best_effort(cfg_.scale_csv));
}

void ScreenCapService::publish_dynamic_state() {
  std::lock_guard<std::mutex> lock(state_mu_);

  auto set_if_changed = [&](const char* field, const json& v) {
    auto it = published_state_.find(field);
    if (it != published_state_.end() && it->second == v)
      return;
    published_state_[field] = v;
    if (bus_)
      f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, v, "tick", json::object());
  };

  set_if_changed("captureRunning", capture_running_.load(std::memory_order_relaxed));
  set_if_changed("lastError", last_error_);

  if (shm_) {
    set_if_changed("videoWidth", shm_->outputWidth());
    set_if_changed("videoHeight", shm_->outputHeight());
    set_if_changed("videoPitch", shm_->outputPitch());
  }
}

void ScreenCapService::request_capture_restart() {
  capture_restart_.store(true, std::memory_order_release);
}

json ScreenCapService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.screencap";
  service["label"] = "Screen Capture";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"video", "capture", "shm"});
  service["stateFields"] = json::array({
      state_field("videoShmName", schema_string(), "ro", "Video SHM", "Shared memory region name", true),
      state_field("videoShmEvent", schema_string(), "ro", "Video Event", "Shared memory event name", false),
      state_field("mode", schema_string_enum({"display", "window", "region"}), "rw", "Mode", "display|window|region",
                  false),
      state_field("fps", schema_number(), "rw", "FPS", "Capture rate", true),
      state_field("displayId", schema_integer(), "ro", "Display ID", "0..N-1 (see listDisplays)", false),
      state_field("windowId", schema_string(), "ro", "Window ID",
                  "backend-specific (e.g. win32:hwnd:0x... or x11:win:0x...)"),
      state_field("window",
                  schema_object(json{{"backend", schema_string()},
                                     {"id", schema_string()},
                                     {"pid", schema_integer()},
                                     {"title", schema_string()},
                                     {"rect", rect_schema()}}),
                  "ro", "Window", "Resolved window metadata (best-effort)", false),
      state_field("region", rect_schema(), "ro", "Region", "Virtual desktop coordinates", false),
      state_field("scale", size_schema(), "ro", "Scale", "Optional output size (0 disables)", false),
      state_field("captureRunning", schema_boolean(), "ro", "Capture Running", "Is capture currently running", false),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message", false),
      state_field("videoWidth", schema_integer(), "ro", "Video Width", "Width of the video frame in pixels", true),
      state_field("videoHeight", schema_integer(), "ro", "Video Height", "Height of the video frame in pixels", true),
      state_field("videoPitch", schema_integer(), "ro", "Video Pitch", "Number of bytes per row of the video frame",
                  false),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array({
      json{{"name", "listDisplays"},
           {"description", "List displays/monitors (backend-specific)"},
           {"showOnNode", false}},
      json{{"name", "pickDisplay"},
           {"description", "Interactive pick a display (hover highlight + click)"},
           {"showOnNode", true}},
      json{{"name", "pickWindow"},
           {"description", "Interactive pick a window (hover highlight + click)"},
           {"showOnNode", true}},
      json{{"name", "pickRegion"},
           {"description", "Interactive pick a region (click-drag to draw)"},
           {"showOnNode", true}},
  });
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array();
  service["dataOutPorts"] = json::array({
      json{{"name", "frameId"},
           {"valueSchema", schema_integer()},
           {"description", "Monotonic frame counter."},
           {"required", false},
           {"showOnNode", false}},
  });
  service["editableDataInPorts"] = false;
  service["editableDataOutPorts"] = false;

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::screencap
