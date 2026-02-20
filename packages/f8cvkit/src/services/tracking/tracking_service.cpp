#include "tracking_service.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <utility>
#include <vector>

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
json schema_string_enum(const std::vector<std::string>& values, const std::string& default_value) {
  json s{{"type", "string"}};
  s["enum"] = json::array();
  for (const std::string& v : values) {
    s["enum"].push_back(v);
  }
  s["default"] = default_value;
  return s;
}
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

json schema_any() { return json{{"type", "any"}}; }

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

bool json_number_to_int(const json& v, int& out) {
  if (!v.is_number()) return false;
  if (v.is_number_integer()) {
    out = v.get<int>();
    return true;
  }
  if (v.is_number_unsigned()) {
    out = static_cast<int>(v.get<unsigned int>());
    return true;
  }
  if (v.is_number_float()) {
    out = static_cast<int>(std::lround(v.get<double>()));
    return true;
  }
  return false;
}

bool json_number_to_double(const json& v, double& out) {
  if (!v.is_number()) return false;
  if (v.is_number_float()) {
    out = v.get<double>();
    return true;
  }
  if (v.is_number_integer()) {
    out = static_cast<double>(v.get<int>());
    return true;
  }
  if (v.is_number_unsigned()) {
    out = static_cast<double>(v.get<unsigned int>());
    return true;
  }
  return false;
}

std::optional<double> extract_score_from_object(const json& obj) {
  if (!obj.is_object()) return std::nullopt;
  static const std::array<const char*, 5> kScoreKeys = {"score", "conf", "confidence", "probability", "prob"};
  for (const char* key : kScoreKeys) {
    if (!obj.contains(key)) continue;
    double s = 0.0;
    if (json_number_to_double(obj.at(key), s)) return s;
  }
  return std::nullopt;
}

bool rect_from_xywh_values(const json& x_v, const json& y_v, const json& w_v, const json& h_v, cv::Rect& out) {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  if (!json_number_to_int(x_v, x) || !json_number_to_int(y_v, y) || !json_number_to_int(w_v, w) ||
      !json_number_to_int(h_v, h)) {
    return false;
  }
  if (w <= 0 || h <= 0) return false;
  out = cv::Rect(x, y, w, h);
  return true;
}

bool rect_from_xyxy_values(const json& x1_v, const json& y1_v, const json& x2_v, const json& y2_v, cv::Rect& out) {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;
  if (!json_number_to_int(x1_v, x1) || !json_number_to_int(y1_v, y1) || !json_number_to_int(x2_v, x2) ||
      !json_number_to_int(y2_v, y2)) {
    return false;
  }
  const int w = x2 - x1;
  const int h = y2 - y1;
  if (w <= 0 || h <= 0) return false;
  out = cv::Rect(x1, y1, w, h);
  return true;
}

bool try_extract_candidate_from_object(const json& obj, TrackingInitCandidate& out) {
  if (!obj.is_object()) return false;

  std::optional<double> score = extract_score_from_object(obj);
  cv::Rect rect;
  if (obj.contains("x") && obj.contains("y") && obj.contains("w") && obj.contains("h")) {
    if (rect_from_xywh_values(obj.at("x"), obj.at("y"), obj.at("w"), obj.at("h"), rect)) {
      out.bbox = rect;
      out.score = score;
      return true;
    }
  }
  if (obj.contains("x1") && obj.contains("y1") && obj.contains("x2") && obj.contains("y2")) {
    if (rect_from_xyxy_values(obj.at("x1"), obj.at("y1"), obj.at("x2"), obj.at("y2"), rect)) {
      out.bbox = rect;
      out.score = score;
      return true;
    }
  }
  if (obj.contains("left") && obj.contains("top") && obj.contains("right") && obj.contains("bottom")) {
    if (rect_from_xyxy_values(obj.at("left"), obj.at("top"), obj.at("right"), obj.at("bottom"), rect)) {
      out.bbox = rect;
      out.score = score;
      return true;
    }
  }
  if (obj.contains("bbox")) {
    const json& bbox = obj.at("bbox");
    if (bbox.is_array() && bbox.size() >= 4) {
      if (rect_from_xyxy_values(bbox.at(0), bbox.at(1), bbox.at(2), bbox.at(3), rect)) {
        out.bbox = rect;
        out.score = score;
        return true;
      }
    }
    if (bbox.is_object()) {
      if (try_extract_candidate_from_object(bbox, out)) {
        if (score.has_value()) out.score = score;
        return true;
      }
    }
  }
  return false;
}

void collect_bbox_candidates(const json& root, std::vector<TrackingInitCandidate>& out, int depth) {
  if (depth > 24 || out.size() >= 256) return;

  TrackingInitCandidate candidate;
  if (try_extract_candidate_from_object(root, candidate)) {
    out.push_back(candidate);
    if (out.size() >= 256) return;
  }

  if (root.is_array()) {
    for (const auto& item : root) {
      collect_bbox_candidates(item, out, depth + 1);
      if (out.size() >= 256) return;
    }
    return;
  }
  if (root.is_object()) {
    for (auto it = root.begin(); it != root.end(); ++it) {
      collect_bbox_candidates(it.value(), out, depth + 1);
      if (out.size() >= 256) return;
    }
  }
}

TrackingInitSelectMode parse_init_select_mode(const std::string& raw, std::string& normalized, bool& ok) {
  std::string s = raw;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  if (s.empty() || s == "closest_center" || s == "closest" || s == "center") {
    normalized = "closest_center";
    ok = true;
    return TrackingInitSelectMode::ClosestCenter;
  }
  if (s == "largest_area" || s == "largest" || s == "area") {
    normalized = "largest_area";
    ok = true;
    return TrackingInitSelectMode::LargestArea;
  }
  if (s == "highest_score" || s == "score") {
    normalized = "highest_score";
    ok = true;
    return TrackingInitSelectMode::HighestScore;
  }
  normalized = "closest_center";
  ok = false;
  return TrackingInitSelectMode::ClosestCenter;
}

std::optional<cv::Rect> pick_best_bbox(
    const std::vector<TrackingInitCandidate>& candidates,
    const cv::Rect& frame_rect,
    TrackingInitSelectMode mode) {
  if (candidates.empty()) return std::nullopt;
  const double cx = static_cast<double>(frame_rect.x) + static_cast<double>(frame_rect.width) * 0.5;
  const double cy = static_cast<double>(frame_rect.y) + static_cast<double>(frame_rect.height) * 0.5;

  bool found = false;
  bool found_scored = false;
  double best_d2 = std::numeric_limits<double>::infinity();
  double best_score = -std::numeric_limits<double>::infinity();
  int best_area = -1;
  cv::Rect best;
  for (const TrackingInitCandidate& candidate : candidates) {
    const cv::Rect clamped = candidate.bbox & frame_rect;
    if (clamped.width <= 0 || clamped.height <= 0) continue;
    const double bx = static_cast<double>(clamped.x) + static_cast<double>(clamped.width) * 0.5;
    const double by = static_cast<double>(clamped.y) + static_cast<double>(clamped.height) * 0.5;
    const double dx = bx - cx;
    const double dy = by - cy;
    const double d2 = dx * dx + dy * dy;
    const int area = clamped.area();

    if (mode == TrackingInitSelectMode::LargestArea) {
      if (!found || area > best_area || (area == best_area && d2 < best_d2)) {
        found = true;
        best_area = area;
        best_d2 = d2;
        best = clamped;
      }
      continue;
    }
    if (mode == TrackingInitSelectMode::HighestScore) {
      if (candidate.score.has_value()) {
        const double score = candidate.score.value();
        if (!found_scored || score > best_score || (score == best_score && d2 < best_d2)) {
          found = true;
          found_scored = true;
          best_score = score;
          best_d2 = d2;
          best = clamped;
        }
        continue;
      }
      if (found_scored) {
        continue;
      }
      if (!found || d2 < best_d2) {
        found = true;
        best_d2 = d2;
        best = clamped;
      }
      continue;
    }

    if (!found || d2 < best_d2) {
      found = true;
      best_d2 = d2;
      best = clamped;
    }
  }
  if (!found) return std::nullopt;
  return best;
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
  bus_->add_command_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  shm_name_override_.clear();
  init_select_mode_ = TrackingInitSelectMode::ClosestCenter;
  init_select_state_ = "closest_center";

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("shmName", "", "init", json::object());
  publish_state_if_changed("initSelect", init_select_state_, "init", json::object());
  publish_state_if_changed("isTracking", false, "init", json::object());
  publish_state_if_changed("isNotTracking", true, "init", json::object());
  publish_state_if_changed("lastError", "", "init", json::object());

  video_.close();
  frame_bgra_.clear();
  last_header_.reset();
  last_frame_id_ = 0;
  last_video_open_attempt_ms_ = 0;

  tracker_.release();
  bbox_ = cv::Rect();
  is_tracking_ = false;
  pending_init_boxes_.clear();

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
  (void)meta;
}

void TrackingService::on_state(const std::string& node_id, const std::string& field, const json& value,
                               std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;
  if (field == "shmName" && value.is_string()) {
    set_shm_name(value.get<std::string>(), meta);
    return;
  }
  if (field == "initSelect" && value.is_string()) {
    set_init_select(value.get<std::string>(), meta);
    return;
  }
}

void TrackingService::on_data(const std::string& node_id, const std::string& port, const json& value,
                              std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  (void)meta;
  if (node_id != cfg_.service_id) return;
  if (port != "initBox") return;
  std::vector<TrackingInitCandidate> candidates;
  collect_bbox_candidates(value, candidates, 0);
  if (candidates.empty()) return;

  {
    std::lock_guard<std::mutex> lock(tracking_mu_);

    // Only accept init boxes when not tracking; tracking should disable matching.
    if (is_tracking_) {
      return;
    }

    pending_init_boxes_ = std::move(candidates);
  }
}

bool TrackingService::on_command(const std::string& call, const json& args, const json& meta, json& result,
                                 std::string& error_code, std::string& error_message) {
  (void)args;
  error_code.clear();
  error_message.clear();
  result = json::object();

  if (call == "stopTracking") {
    json tracking_meta = json::object();
    if (meta.is_object()) tracking_meta = meta;
    tracking_meta["source"] = "command";
    tracking_meta["call"] = call;

    bool was_tracking = false;
    {
      std::lock_guard<std::mutex> lock(tracking_mu_);
      was_tracking = is_tracking_;
      stop_tracking_internal(tracking_meta);
    }

    result["stopped"] = true;
    result["wasTracking"] = was_tracking;
    return true;
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void TrackingService::stop_tracking_internal(const json& meta) {
  tracker_.release();
  bbox_ = cv::Rect();
  pending_init_boxes_.clear();
  set_tracking(false, meta);
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

void TrackingService::set_init_select(const std::string& mode, const json& meta) {
  std::string normalized;
  bool ok = false;
  const TrackingInitSelectMode parsed = parse_init_select_mode(mode, normalized, ok);
  if (!ok) {
    publish_state_if_changed("lastError", "invalid initSelect: " + mode, "state", meta);
    return;
  }
  init_select_mode_ = parsed;
  init_select_state_ = normalized;
  publish_state_if_changed("initSelect", init_select_state_, "state", meta);
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
  std::vector<TrackingInitCandidate> candidates;
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (is_tracking_) return;
    if (pending_init_boxes_.empty()) return;
    candidates = pending_init_boxes_;
    pending_init_boxes_.clear();
  }

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
  try {
    cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);
  } catch (const cv::Exception& ex) {
    publish_state_if_changed("lastError", std::string("opencv cvtColor failed: ") + ex.what(), "runtime", json::object());
    return;
  }

  // Clamp to frame and pick center-nearest bbox.
  cv::Rect frame_rect(0, 0, static_cast<int>(hdr.width), static_cast<int>(hdr.height));
  const std::optional<cv::Rect> selected = pick_best_bbox(candidates, frame_rect, init_select_mode_);
  if (!selected.has_value()) {
    publish_state_if_changed("lastError", "initBox has no valid bbox candidate", "runtime", json::object());
    return;
  }
  cv::Rect bb = selected.value();

  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (is_tracking_) return;
    try {
      tracker_ = cv::TrackerCSRT::create();
      if (tracker_.empty()) {
        publish_state_if_changed("lastError", "TrackerCSRT::create failed", "runtime", json::object());
        return;
      }
      tracker_->init(bgr, bb);
      bbox_ = bb;
      set_tracking(true, json::object({{"source", "initBox"}, {"candidates", static_cast<int>(candidates.size())}}));
    } catch (const cv::Exception& ex) {
      publish_state_if_changed("lastError", std::string("opencv tracker init failed: ") + ex.what(), "runtime",
                               json::object({{"source", "initBox"}}));
      stop_tracking_internal(json::object({{"reason", "opencv_exception"}, {"source", "initBox"}}));
      return;
    } catch (const std::exception& ex) {
      publish_state_if_changed("lastError", std::string("tracker init failed: ") + ex.what(), "runtime",
                               json::object({{"source", "initBox"}}));
      stop_tracking_internal(json::object({{"reason", "std_exception"}, {"source", "initBox"}}));
      return;
    }
  }
}

void TrackingService::process_frame_once() {
  cv::Ptr<cv::Tracker> tracker;
  cv::Rect bbox;
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (!is_tracking_ || tracker_.empty()) {
      // Not tracking: emit a minimal status so downstream can read isTracking/isNotTracking.
      return;
    }
    tracker = tracker_;
    bbox = bbox_;
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
  try {
    cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);
  } catch (const cv::Exception& ex) {
    publish_state_if_changed("lastError", std::string("opencv cvtColor failed: ") + ex.what(), "runtime", json::object());
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "opencv_exception"}, {"where", "cvtColor"}}));
    return;
  }

  cv::Rect out_bbox = bbox;
  bool ok = false;
  try {
    ok = tracker->update(bgr, out_bbox);
  } catch (const cv::Exception& ex) {
    publish_state_if_changed("lastError", std::string("opencv tracker update failed: ") + ex.what(), "runtime",
                             json::object());
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "opencv_exception"}, {"where", "update"}}));
    return;
  } catch (const std::exception& ex) {
    publish_state_if_changed("lastError", std::string("tracker update failed: ") + ex.what(), "runtime", json::object());
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "std_exception"}, {"where", "update"}}));
    return;
  }
  if (!ok) {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "update_failed"}}));
    return;
  }
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (!is_tracking_) return;
    bbox_ = out_bbox;
  }
  const cv::Rect emit_bbox = out_bbox;

  json out = json::object();
  out["frameId"] = hdr.frame_id;
  out["tsMs"] = hdr.ts_ms;
  out["width"] = hdr.width;
  out["height"] = hdr.height;
  out["status"] = "tracking";
  out["bbox"] = json::array(
      {emit_bbox.x, emit_bbox.y, emit_bbox.x + emit_bbox.width, emit_bbox.y + emit_bbox.height});
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
  const json init_box_schema = schema_any();

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
      state_field("shmName", schema_string(), "rw", "Video SHM", "Optional SHM name override (e.g. shm.xxx.video).", true),
      state_field("initSelect",
                  schema_string_enum({"closest_center", "largest_area", "highest_score"}, "closest_center"),
                  "rw", "Init Select",
                  "Init bbox selection strategy: closest_center | largest_area | highest_score.", true),
      state_field("isTracking", schema_boolean(), "ro", "Is Tracking", "True when tracker is running.", true),
      state_field("isNotTracking", schema_boolean(), "ro", "Is Not Tracking", "Negation of isTracking.", true),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message.", true),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array({
      json{{"name", "stopTracking"},
           {"description", "Stop current tracking and return to waiting for initBox."},
           {"showOnNode", true}},
  });
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array({
      json{{"name", "initBox"},
           {"valueSchema", init_box_schema},
           {"description",
            "Init payload (single bbox or nested detection tree). Recursively extracts bbox candidates and uses the one "
            "closest to image center."},
           {"required", false}},
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
