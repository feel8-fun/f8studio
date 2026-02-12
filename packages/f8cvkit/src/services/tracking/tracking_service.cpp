#include "tracking_service.h"

#include <cctype>
#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/shm/naming.h"
#include "f8cppsdk/shm/sizing.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cvkit::tracking {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_number() { return json{{"type", "number"}}; }
json schema_integer() { return json{{"type", "integer"}}; }
json schema_boolean() { return json{{"type", "boolean"}}; }

json schema_object(const json& props, const json& required = json::array()) {
  json obj;
  obj["type"] = "object";
  obj["properties"] = props;
  if (required.is_array()) obj["required"] = required;
  obj["additionalProperties"] = false;
  return obj;
}

json schema_array(const json& item_schema) {
  json arr;
  arr["type"] = "array";
  arr["items"] = item_schema;
  return arr;
}

json state_field(std::string name, const json& value_schema, std::string access, std::string label = {},
                 std::string description = {}, bool show_on_node = false, std::string ui_control = {}) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  if (!label.empty()) sf["label"] = std::move(label);
  if (!description.empty()) sf["description"] = std::move(description);
  if (show_on_node) sf["showOnNode"] = true;
  if (!ui_control.empty()) sf["uiControl"] = std::move(ui_control);
  return sf;
}

}  // namespace

TrackingService::TrackingService(Config cfg) : cfg_(std::move(cfg)) {}

TrackingService::~TrackingService() { stop(); }

bool TrackingService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Tracking";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("active", active_.load(std::memory_order_acquire), "init", json::object());
  publish_state_if_changed("shmName", "", "init", json::object());
  publish_state_if_changed("isTracking", false, "init", json::object());
  publish_state_if_changed("isNotTracking", true, "init", json::object());
  publish_state_if_changed("lastError", "", "init", json::object());

  shm_name_override_.clear();

  video_.close();
  frame_bgra_.clear();
  last_header_.reset();
  last_frame_id_ = 0;
  last_video_open_attempt_ms_ = 0;

  tracker_.release();
  bbox_ = cv::Rect();
  is_tracking_ = false;
  pending_init_box_.reset();

  if (!cfg_.shm_name.empty()) {
    set_shm_name(cfg_.shm_name, json::object({{"init", true}}));
  }

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_tracking started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void TrackingService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (bus_) bus_->stop();
  bus_.reset();
}

void TrackingService::tick() {
  if (!running()) return;
  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
  }

  if (!active_.load(std::memory_order_acquire)) {
    return;
  }

  apply_init_box_if_any();
  process_frame_once();
}

void TrackingService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                               const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) {
    (void)f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, value, source, meta);
  }
}

void TrackingService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  publish_state_if_changed("active", active, "lifecycle", meta);
}

void TrackingService::on_state(const std::string& node_id, const std::string& field, const json& value,
                               std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;
  if (field == "active" && value.is_boolean()) {
    active_.store(value.get<bool>(), std::memory_order_release);
    publish_state_if_changed("active", active_.load(std::memory_order_acquire), "state", json::object());
    return;
  }
  if (field == "shmName" && value.is_string()) {
    set_shm_name(value.get<std::string>(), meta);
    return;
  }
}

void TrackingService::on_data(const std::string& node_id, const std::string& port, const json& value,
                              std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;
  if (port != "initBox") return;
  if (!value.is_object()) return;

  const int x = value.value("x", 0);
  const int y = value.value("y", 0);
  const int w = value.value("w", 0);
  const int h = value.value("h", 0);
  if (w <= 0 || h <= 0) return;

  // Only accept init boxes when not tracking; tracking should disable matching.
  if (is_tracking_) {
    return;
  }

  pending_init_box_ = cv::Rect(x, y, w, h);
}

void TrackingService::set_shm_name(const std::string& shm_name, const json& meta) {
  std::string s = shm_name;
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();

  if (s == shm_name_override_) {
    publish_state_if_changed("shmName", shm_name_override_, "state", meta);
    return;
  }
  shm_name_override_ = s;
  publish_state_if_changed("shmName", shm_name_override_, "state", meta);
  video_.close();
  last_video_open_attempt_ms_ = 0;
}

bool TrackingService::ensure_video_open() {
  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (video_.readHeader(hdr)) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_video_open_attempt_ms_ > 0 && (now - last_video_open_attempt_ms_) < 1000) {
    return false;
  }
  last_video_open_attempt_ms_ = now;

  std::string shm_name = shm_name_override_;
  if (shm_name.empty()) {
    shm_name = f8::cppsdk::shm::video_shm_name(cfg_.service_id);
  }

  const std::size_t bytes = f8::cppsdk::shm::kDefaultVideoShmBytes;
  if (!video_.open(shm_name, bytes)) {
    publish_state_if_changed("lastError", "video shm open failed: " + shm_name, "runtime", json::object());
    return false;
  }
  publish_state_if_changed("lastError", "", "runtime", json::object());
  return true;
}

void TrackingService::apply_init_box_if_any() {
  if (is_tracking_) return;
  if (!pending_init_box_.has_value()) return;

  const auto bbox = pending_init_box_.value();
  pending_init_box_.reset();

  if (!ensure_video_open()) return;

  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (!video_.copyLatestFrame(frame_bgra_, hdr)) {
    publish_state_if_changed("lastError", "failed to read video frame for init", "runtime", json::object());
    return;
  }
  if (hdr.format != 1 || hdr.width == 0 || hdr.height == 0 || hdr.pitch == 0) {
    publish_state_if_changed("lastError", "unsupported video shm format", "runtime", json::object());
    return;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(hdr.pitch);
  if (frame_bgra_.size() < row_bytes * static_cast<std::size_t>(hdr.height)) {
    publish_state_if_changed("lastError", "video shm frame too small", "runtime", json::object());
    return;
  }

  cv::Mat bgra_mat(static_cast<int>(hdr.height), static_cast<int>(hdr.width), CV_8UC4,
                   const_cast<std::byte*>(frame_bgra_.data()), static_cast<std::size_t>(hdr.pitch));
  cv::Mat bgr;
  cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);

  // Clamp bbox to frame.
  cv::Rect frame_rect(0, 0, static_cast<int>(hdr.width), static_cast<int>(hdr.height));
  cv::Rect bb = bbox & frame_rect;
  if (bb.width <= 0 || bb.height <= 0) {
    publish_state_if_changed("lastError", "initBox out of bounds", "runtime", json::object());
    return;
  }

  tracker_ = cv::TrackerCSRT::create();
  if (tracker_.empty()) {
    publish_state_if_changed("lastError", "TrackerCSRT::create failed", "runtime", json::object());
    return;
  }
  tracker_->init(bgr, bb);
  bbox_ = bb;

  set_tracking(true, json::object({{"source", "initBox"}}));
}

void TrackingService::process_frame_once() {
  if (!is_tracking_ || tracker_.empty()) {
    // Not tracking: emit a minimal status so downstream can read isTracking/isNotTracking.
    return;
  }
  if (!ensure_video_open()) {
    return;
  }

  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (!video_.copyLatestFrame(frame_bgra_, hdr)) {
    return;
  }
  if (hdr.frame_id == 0 || hdr.frame_id == last_frame_id_) {
    return;
  }
  last_frame_id_ = hdr.frame_id;
  last_header_ = hdr;

  if (hdr.format != 1 || hdr.width == 0 || hdr.height == 0 || hdr.pitch == 0) {
    publish_state_if_changed("lastError", "unsupported video shm format", "runtime", json::object());
    set_tracking(false, json::object({{"reason", "bad_format"}}));
    return;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(hdr.pitch);
  if (frame_bgra_.size() < row_bytes * static_cast<std::size_t>(hdr.height)) {
    publish_state_if_changed("lastError", "video shm frame too small", "runtime", json::object());
    set_tracking(false, json::object({{"reason", "bad_frame"}}));
    return;
  }

  cv::Mat bgra_mat(static_cast<int>(hdr.height), static_cast<int>(hdr.width), CV_8UC4,
                   const_cast<std::byte*>(frame_bgra_.data()), static_cast<std::size_t>(hdr.pitch));
  cv::Mat bgr;
  cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);

  cv::Rect out_bbox = bbox_;
  const bool ok = tracker_->update(bgr, out_bbox);
  if (!ok) {
    set_tracking(false, json::object({{"reason", "update_failed"}}));
    return;
  }
  bbox_ = out_bbox;

  json out = json::object();
  out["frameId"] = hdr.frame_id;
  out["tsMs"] = hdr.ts_ms;
  out["width"] = hdr.width;
  out["height"] = hdr.height;
  out["status"] = "tracking";
  out["bbox"] = json::array({bbox_.x, bbox_.y, bbox_.x + bbox_.width, bbox_.y + bbox_.height});
  out["tracker"] = json::object({{"kind", "csrt"}, {"ok", true}});

  publish_state_if_changed("lastError", "", "runtime", json::object());
  if (bus_) {
    (void)bus_->emit_data(cfg_.service_id, "tracking", out);
  }
}

void TrackingService::set_tracking(bool tracking, const json& meta) {
  if (tracking == is_tracking_) return;
  is_tracking_ = tracking;
  publish_state_if_changed("isTracking", is_tracking_, "runtime", meta);
  publish_state_if_changed("isNotTracking", !is_tracking_, "runtime", meta);
}

json TrackingService::describe() {
  const json init_box_schema =
      schema_object(json{{"x", schema_integer()},
                         {"y", schema_integer()},
                         {"w", schema_integer()},
                         {"h", schema_integer()},
                         {"score", schema_number()},
                         {"frameId", schema_integer()},
                         {"tsMs", schema_integer()}});

  const json tracking_schema = schema_object(
      json{{"frameId", schema_integer()},
           {"tsMs", schema_integer()},
           {"width", schema_integer()},
           {"height", schema_integer()},
           {"status", schema_string()},
           {"bbox", schema_array(schema_integer())},
           {"tracker", schema_object(json{{"kind", schema_string()}, {"ok", schema_boolean()}})}});

  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.tracking";
  service["label"] = "CVKit Tracking";
  service["version"] = "0.0.1";
  service["rendererClass"] = "defaultService";
  service["tags"] = json::array({"cv", "tracking"});
  service["stateFields"] = json::array({
      state_field("active", schema_boolean(), "rw", "Active", "Enable/disable tracking.", true),
      state_field("shmName", schema_string(), "rw", "SHM Name", "Optional SHM name override (e.g. shm.xxx.video).", true),
      state_field("isTracking", schema_boolean(), "ro", "Is Tracking", "True when tracker is running."),
      state_field("isNotTracking", schema_boolean(), "ro", "Is Not Tracking", "Negation of isTracking.", true),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message."),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array();
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array({
      json{{"name", "initBox"}, {"valueSchema", init_box_schema}, {"description", "Init box stream."}, {"required", false}},
  });
  service["dataOutPorts"] = json::array({
      json{{"name", "tracking"}, {"valueSchema", tracking_schema}, {"description", "Tracking output stream."}, {"required", false}},
  });
  service["editableDataInPorts"] = false;
  service["editableDataOutPorts"] = false;

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::tracking
