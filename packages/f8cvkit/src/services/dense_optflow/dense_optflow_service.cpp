#include "dense_optflow_service.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/shm/naming.h"
#include "f8cppsdk/shm/sizing.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cvkit::dense_optflow {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_number() { return json{{"type", "number"}}; }
json schema_integer() { return json{{"type", "integer"}}; }

json schema_number(double default_value, double minimum, double maximum) {
  json s{{"type", "number"}};
  s["default"] = default_value;
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
}

json schema_integer(int default_value, int minimum, int maximum) {
  json s{{"type", "integer"}};
  s["default"] = default_value;
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
}

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

std::string trim_copy(const std::string& value) {
  std::string out = value;
  while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
  while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
  return out;
}

bool parse_int_value(const json& value, int& out) {
  if (value.is_number_integer()) {
    out = value.get<int>();
    return true;
  }
  if (value.is_number_unsigned()) {
    out = static_cast<int>(value.get<unsigned int>());
    return true;
  }
  if (value.is_number_float()) {
    out = static_cast<int>(std::lround(value.get<double>()));
    return true;
  }
  return false;
}

bool parse_double_value(const json& value, double& out) {
  if (value.is_number_float()) {
    out = value.get<double>();
    return true;
  }
  if (value.is_number_integer()) {
    out = static_cast<double>(value.get<int>());
    return true;
  }
  if (value.is_number_unsigned()) {
    out = static_cast<double>(value.get<unsigned int>());
    return true;
  }
  return false;
}

}  // namespace

DenseOptflowService::DenseOptflowService(Config cfg) : cfg_(std::move(cfg)) {}

DenseOptflowService::~DenseOptflowService() { stop(); }

bool DenseOptflowService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Dense Optical Flow";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  input_shm_name_.clear();
  compute_every_n_frames_ = 2;
  sample_step_px_ = 16;
  min_mag_ = 0.0;

  video_.close();
  frame_bgra_.clear();
  last_notify_seq_ = 0;
  last_frame_id_ = 0;
  last_video_open_attempt_ms_ = 0;
  frame_counter_ = 0;

  prev_gray_.release();
  has_prev_gray_ = false;
  prev_width_ = 0;
  prev_height_ = 0;

  telemetry_observed_frames_ = 0;
  telemetry_processed_frames_ = 0;
  telemetry_window_processed_frames_ = 0;
  telemetry_fail_frames_ = 0;
  telemetry_last_vectors_per_frame_ = 0;
  telemetry_window_start_ms_ = 0;
  telemetry_last_process_ms_ = 0.0;
  telemetry_total_process_ms_ = 0.0;
  telemetry_fps_ = 0.0;

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("inputShmName", "", "init", json::object());
  publish_state_if_changed("computeEveryNFrames", compute_every_n_frames_, "init", json::object());
  publish_state_if_changed("sampleStepPx", sample_step_px_, "init", json::object());
  publish_state_if_changed("minMag", min_mag_, "init", json::object());
  publish_state_if_changed("lastError", "", "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_dense_optflow started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void DenseOptflowService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();

  std::lock_guard<std::mutex> lock(flow_mu_);
  video_.close();
}

void DenseOptflowService::tick() {
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
  process_frame_once();
}

void DenseOptflowService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                                   const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) {
    (void)f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, value, source, meta);
  }
}

void DenseOptflowService::emit_telemetry(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms,
                                         std::uint64_t vectors_per_frame) {
  if (!bus_) return;
  if (telemetry_window_start_ms_ <= 0) {
    telemetry_window_start_ms_ = ts_ms;
  }
  ++telemetry_processed_frames_;
  ++telemetry_window_processed_frames_;
  telemetry_last_process_ms_ = process_ms;
  telemetry_total_process_ms_ += process_ms;
  telemetry_last_vectors_per_frame_ = vectors_per_frame;

  const std::int64_t elapsed = ts_ms - telemetry_window_start_ms_;
  if (elapsed >= 1000) {
    telemetry_fps_ = static_cast<double>(telemetry_window_processed_frames_) * 1000.0 / static_cast<double>(elapsed);
    telemetry_window_start_ms_ = ts_ms;
    telemetry_window_processed_frames_ = 0;
  }

  const std::uint64_t dropped_frames = telemetry_observed_frames_ > telemetry_processed_frames_
                                           ? (telemetry_observed_frames_ - telemetry_processed_frames_)
                                           : 0;
  const double avg_process_ms = telemetry_processed_frames_ > 0
                                    ? (telemetry_total_process_ms_ / static_cast<double>(telemetry_processed_frames_))
                                    : 0.0;

  json telemetry = json::object();
  telemetry["tsMs"] = ts_ms;
  telemetry["frameId"] = frame_id;
  telemetry["fps"] = telemetry_fps_;
  telemetry["processMs"] = telemetry_last_process_ms_;
  telemetry["avgProcessMs"] = avg_process_ms;
  telemetry["observedFrames"] = telemetry_observed_frames_;
  telemetry["processedFrames"] = telemetry_processed_frames_;
  telemetry["droppedFrames"] = dropped_frames;
  telemetry["vectorsPerFrame"] = telemetry_last_vectors_per_frame_;
  telemetry["failFrames"] = telemetry_fail_frames_;
  (void)bus_->emit_data(cfg_.service_id, "telemetry", telemetry);
}

void DenseOptflowService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

void DenseOptflowService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                   std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;

  if (field == "inputShmName" && value.is_string()) {
    const std::string next = trim_copy(value.get<std::string>());
    {
      std::lock_guard<std::mutex> lock(flow_mu_);
      if (next == input_shm_name_) {
        publish_state_if_changed("inputShmName", input_shm_name_, "state", meta);
        return;
      }
      input_shm_name_ = next;
      video_.close();
      last_video_open_attempt_ms_ = 0;
      last_notify_seq_ = 0;
      last_frame_id_ = 0;
      frame_counter_ = 0;
      frame_bgra_.clear();
      prev_gray_.release();
      has_prev_gray_ = false;
      prev_width_ = 0;
      prev_height_ = 0;
    }
    publish_state_if_changed("inputShmName", input_shm_name_, "state", meta);
    return;
  }

  if (field == "computeEveryNFrames") {
    int v = 0;
    if (!parse_int_value(value, v)) {
      publish_state_if_changed("lastError", "invalid computeEveryNFrames", "state", meta);
      return;
    }
    v = std::max(1, std::min(120, v));
    compute_every_n_frames_ = v;
    publish_state_if_changed("computeEveryNFrames", compute_every_n_frames_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "sampleStepPx") {
    int v = 0;
    if (!parse_int_value(value, v)) {
      publish_state_if_changed("lastError", "invalid sampleStepPx", "state", meta);
      return;
    }
    v = std::max(4, std::min(128, v));
    sample_step_px_ = v;
    publish_state_if_changed("sampleStepPx", sample_step_px_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "minMag") {
    double v = 0.0;
    if (!parse_double_value(value, v)) {
      publish_state_if_changed("lastError", "invalid minMag", "state", meta);
      return;
    }
    v = std::max(0.0, std::min(100.0, v));
    min_mag_ = v;
    publish_state_if_changed("minMag", min_mag_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }
}

void DenseOptflowService::on_data(const std::string& node_id, const std::string& port, const json& value,
                                  std::int64_t ts_ms, const json& meta) {
  (void)node_id;
  (void)port;
  (void)value;
  (void)ts_ms;
  (void)meta;
  // SHM pull mode only.
}

bool DenseOptflowService::ensure_video_open() {
  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (video_.readHeader(hdr)) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_video_open_attempt_ms_ > 0 && (now - last_video_open_attempt_ms_) < 1000) {
    return false;
  }
  last_video_open_attempt_ms_ = now;

  if (input_shm_name_.empty()) {
    publish_state_if_changed("lastError", "missing inputShmName", "runtime", json::object());
    return false;
  }

  if (!video_.open(input_shm_name_, f8::cppsdk::shm::kDefaultVideoShmBytes)) {
    publish_state_if_changed("lastError", "video shm open failed: " + input_shm_name_, "runtime", json::object());
    return false;
  }

  last_notify_seq_ = 0;
  publish_state_if_changed("lastError", "", "runtime", json::object());
  return true;
}

void DenseOptflowService::process_frame_once() {
  if (!bus_) return;

  std::lock_guard<std::mutex> lock(flow_mu_);
  if (!ensure_video_open()) {
    return;
  }

  std::uint32_t observed_notify_seq = last_notify_seq_;
  if (!video_.waitNewFrame(last_notify_seq_, 20, &observed_notify_seq)) {
    return;
  }
  last_notify_seq_ = observed_notify_seq;

  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (!video_.copyLatestFrame(frame_bgra_, hdr)) {
    return;
  }
  if (hdr.frame_id == 0 || hdr.frame_id == last_frame_id_) {
    return;
  }

  ++telemetry_observed_frames_;
  ++frame_counter_;
  last_frame_id_ = hdr.frame_id;

  if (hdr.format != 1 || hdr.width == 0 || hdr.height == 0 || hdr.pitch == 0) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "unsupported video shm format", "runtime", json::object());
    return;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(hdr.pitch);
  if (row_bytes < static_cast<std::size_t>(hdr.width) * 4) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "invalid video shm pitch", "runtime", json::object());
    return;
  }
  if (frame_bgra_.size() < row_bytes * static_cast<std::size_t>(hdr.height)) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "video shm frame too small", "runtime", json::object());
    return;
  }

  cv::Mat bgra(static_cast<int>(hdr.height), static_cast<int>(hdr.width), CV_8UC4, const_cast<std::byte*>(frame_bgra_.data()),
               row_bytes);
  cv::Mat gray;
  try {
    cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);
  } catch (const cv::Exception& ex) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", std::string("opencv cvtColor failed: ") + ex.what(), "runtime", json::object());
    return;
  }

  if (!has_prev_gray_ || prev_width_ != static_cast<int>(hdr.width) || prev_height_ != static_cast<int>(hdr.height)) {
    prev_gray_ = gray;
    has_prev_gray_ = true;
    prev_width_ = static_cast<int>(hdr.width);
    prev_height_ = static_cast<int>(hdr.height);
    return;
  }

  if ((frame_counter_ % static_cast<std::uint64_t>(std::max(1, compute_every_n_frames_))) != 0) {
    return;
  }

  const std::int64_t process_start_ms = f8::cppsdk::now_ms();

  cv::Mat flow;
  try {
    cv::calcOpticalFlowFarneback(prev_gray_, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
  } catch (const cv::Exception& ex) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", std::string("opencv farneback failed: ") + ex.what(), "runtime", json::object());
    prev_gray_ = gray;
    return;
  }

  std::uint64_t grid_points = 0;
  std::uint64_t kept_points = 0;
  double mag_sum = 0.0;
  double max_mag = 0.0;

  json vectors = json::array();
  const int step = std::max(4, sample_step_px_);
  for (int y = step / 2; y < flow.rows; y += step) {
    for (int x = step / 2; x < flow.cols; x += step) {
      ++grid_points;
      const cv::Point2f d = flow.at<cv::Point2f>(y, x);
      const double dx = static_cast<double>(d.x);
      const double dy = static_cast<double>(d.y);
      const double mag = std::sqrt(dx * dx + dy * dy);
      if (mag < min_mag_) {
        continue;
      }
      ++kept_points;
      mag_sum += mag;
      if (mag > max_mag) max_mag = mag;

      json vec = json::object();
      vec["x"] = x;
      vec["y"] = y;
      vec["dx"] = dx;
      vec["dy"] = dy;
      vec["mag"] = mag;
      vectors.push_back(std::move(vec));
    }
  }

  const double mean_mag = kept_points > 0 ? (mag_sum / static_cast<double>(kept_points)) : 0.0;

  json out = json::object();
  out["schemaVersion"] = "f8visionFlowField/1";
  out["frameId"] = hdr.frame_id;
  out["tsMs"] = hdr.ts_ms;
  out["width"] = hdr.width;
  out["height"] = hdr.height;
  out["model"] = "farneback";
  out["sampleStepPx"] = step;
  out["vectors"] = std::move(vectors);
  out["stats"] = json::object({
      {"gridPoints", grid_points},
      {"keptPoints", kept_points},
      {"meanMag", mean_mag},
      {"maxMag", max_mag},
  });

  publish_state_if_changed("lastError", "", "runtime", json::object());
  (void)bus_->emit_data(cfg_.service_id, "flowField", out);

  const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
  emit_telemetry(end_ts_ms, hdr.frame_id, static_cast<double>(end_ts_ms - process_start_ms), kept_points);

  prev_gray_ = gray;
}

json DenseOptflowService::describe() {
  const json vector_schema = schema_object(
      json{{"x", schema_number()}, {"y", schema_number()}, {"dx", schema_number()}, {"dy", schema_number()}, {"mag", schema_number()}});
  const json flow_schema = schema_object(
      json{{"schemaVersion", schema_string()},
           {"frameId", schema_integer()},
           {"tsMs", schema_integer()},
           {"width", schema_integer()},
           {"height", schema_integer()},
           {"model", schema_string()},
           {"sampleStepPx", schema_integer()},
           {"vectors", schema_array(vector_schema)},
           {"stats",
            schema_object(json{{"gridPoints", schema_integer()},
                               {"keptPoints", schema_integer()},
                               {"meanMag", schema_number()},
                               {"maxMag", schema_number()}})}});
  const json telemetry_schema = schema_object(
      json{{"tsMs", schema_integer()},
           {"frameId", schema_integer()},
           {"fps", schema_number()},
           {"processMs", schema_number()},
           {"avgProcessMs", schema_number()},
           {"observedFrames", schema_integer()},
           {"processedFrames", schema_integer()},
           {"droppedFrames", schema_integer()},
           {"vectorsPerFrame", schema_integer()},
           {"failFrames", schema_integer()}});

  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.denseoptflow";
  service["label"] = "CVKit Dense Optical Flow";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"cv", "optical_flow", "flow_field"});
  service["stateFields"] = json::array({
      state_field("inputShmName", schema_string(), "rw", "Input Video SHM", "Input SHM name (e.g. shm.xxx.video).", true),
      state_field("computeEveryNFrames", schema_integer(2, 1, 120), "rw", "Compute Every N Frames",
                  "Compute flow once per N new frames.", true),
      state_field("sampleStepPx", schema_integer(16, 4, 128), "rw", "Sample Step (px)",
                  "Grid sampling step in pixels for output vectors.", true),
      state_field("minMag", schema_number(0.0, 0.0, 100.0), "rw", "Min Magnitude",
                  "Drop vectors with magnitude below this threshold.", true),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message.", true),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array();
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array();
  service["dataOutPorts"] = json::array({
      json{{"name", "flowField"},
           {"valueSchema", flow_schema},
           {"description", "Optical flow vectors in schema f8visionFlowField/1."},
           {"required", false}},
      json{{"name", "telemetry"},
           {"valueSchema", telemetry_schema},
           {"description", "Runtime telemetry: fps/process time/vectors/failures."},
           {"required", false}},
  });
  service["editableDataInPorts"] = false;
  service["editableDataOutPorts"] = false;

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::dense_optflow
