#include "dense_optflow_service.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
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

std::uint16_t float32_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xFFu;
  const std::uint32_t mantissa = bits & 0x7FFFFFu;

  if (exponent == 0xFFu) {
    if (mantissa == 0u) return static_cast<std::uint16_t>(sign | 0x7C00u);
    const std::uint16_t nan_payload = static_cast<std::uint16_t>(mantissa >> 13);
    return static_cast<std::uint16_t>(sign | 0x7C00u | (nan_payload ? nan_payload : 1u));
  }

  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    std::uint32_t mant = mantissa | 0x800000u;
    const int shift = 14 - half_exponent;
    std::uint16_t half_mantissa = static_cast<std::uint16_t>(mant >> shift);
    const std::uint32_t lsb = (mant >> (shift - 1)) & 1u;
    const std::uint32_t rest = mant & ((1u << (shift - 1)) - 1u);
    if (lsb != 0u && (rest != 0u || (half_mantissa & 1u) != 0u)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> 13);
  const std::uint32_t lsb = (mantissa >> 12) & 1u;
  const std::uint32_t rest = mantissa & 0xFFFu;
  if (lsb != 0u && (rest != 0u || (half_mantissa & 1u) != 0u)) {
    ++half_mantissa;
    if (half_mantissa == 0x0400u) {
      half_mantissa = 0;
      ++half_exponent;
      if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
      }
    }
  }

  return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(half_exponent) << 10) | half_mantissa);
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
  flow_shm_name_ = "shm." + cfg_.service_id + ".flow";
  flow_shm_format_ = "flow2_f16";
  compute_scale_ = 0.5;

  video_.close();
  frame_bgra_.clear();
  flow_payload_.clear();
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
  publish_state_if_changed("flowShmName", flow_shm_name_, "init", json::object());
  publish_state_if_changed("flowShmFormat", flow_shm_format_, "init", json::object());
  publish_state_if_changed("computeScale", compute_scale_, "init", json::object());
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

  if (field == "computeScale") {
    double v = 0.0;
    if (!parse_double_value(value, v)) {
      publish_state_if_changed("lastError", "invalid computeScale", "state", meta);
      return;
    }
    compute_scale_ = std::max(0.25, std::min(1.0, v));
    publish_state_if_changed("computeScale", compute_scale_, "state", meta);
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

  const double scale = std::max(0.25, std::min(1.0, compute_scale_));
  compute_scale_ = scale;

  cv::Mat prev_compute = prev_gray_;
  cv::Mat gray_compute = gray;
  if (scale < 0.999) {
    int sw = static_cast<int>(std::lround(static_cast<double>(gray.cols) * scale));
    int sh = static_cast<int>(std::lround(static_cast<double>(gray.rows) * scale));
    sw = std::max(sw, 1);
    sh = std::max(sh, 1);
    try {
      cv::resize(prev_gray_, prev_compute, cv::Size(sw, sh), 0.0, 0.0, cv::INTER_AREA);
      cv::resize(gray, gray_compute, cv::Size(sw, sh), 0.0, 0.0, cv::INTER_AREA);
    } catch (const cv::Exception& ex) {
      ++telemetry_fail_frames_;
      publish_state_if_changed("lastError", std::string("opencv resize failed: ") + ex.what(), "runtime", json::object());
      prev_gray_ = gray;
      return;
    }
  }

  cv::Mat flow_compute;
  try {
    cv::calcOpticalFlowFarneback(prev_compute, gray_compute, flow_compute, 0.5, 3, 15, 3, 5, 1.2, 0);
  } catch (const cv::Exception& ex) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", std::string("opencv farneback failed: ") + ex.what(), "runtime", json::object());
    prev_gray_ = gray;
    return;
  }

  cv::Mat flow = flow_compute;
  if (flow_compute.cols != gray.cols || flow_compute.rows != gray.rows) {
    try {
      cv::resize(flow_compute, flow, gray.size(), 0.0, 0.0, cv::INTER_LINEAR);
      flow *= static_cast<float>(1.0 / scale);
    } catch (const cv::Exception& ex) {
      ++telemetry_fail_frames_;
      publish_state_if_changed("lastError", std::string("opencv flow upscale failed: ") + ex.what(), "runtime", json::object());
      prev_gray_ = gray;
      return;
    }
  }

  std::string shm_name = trim_copy(flow_shm_name_);
  if (shm_name.empty()) {
    shm_name = "shm." + cfg_.service_id + ".flow";
    flow_shm_name_ = shm_name;
    publish_state_if_changed("flowShmName", flow_shm_name_, "runtime", json::object());
  }
  if (flow_sink_.regionName() != shm_name) {
    if (!flow_sink_.initialize(shm_name, f8::cppsdk::shm::kDefaultVideoShmBytes, f8::cppsdk::shm::kDefaultVideoShmSlots)) {
      ++telemetry_fail_frames_;
      publish_state_if_changed("lastError", "flow shm init failed: " + shm_name, "runtime", json::object());
      prev_gray_ = gray;
      return;
    }
  }
  if (!flow_sink_.ensureConfigurationForFormat(static_cast<unsigned>(flow.cols), static_cast<unsigned>(flow.rows),
                                               f8::cppsdk::kVideoFormatFlow2F16, 4)) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "flow shm ensureConfiguration failed", "runtime", json::object());
    prev_gray_ = gray;
    return;
  }

  const std::size_t flow_pitch = static_cast<std::size_t>(flow_sink_.outputPitch());
  const std::size_t flow_bytes = flow_pitch * static_cast<std::size_t>(flow.rows);
  flow_payload_.assign(flow_bytes, std::byte{0});
  for (int y = 0; y < flow.rows; ++y) {
    std::byte* row = flow_payload_.data() + static_cast<std::size_t>(y) * flow_pitch;
    for (int x = 0; x < flow.cols; ++x) {
      const cv::Point2f d = flow.at<cv::Point2f>(y, x);
      const std::uint16_t hu = float32_to_half(d.x);
      const std::uint16_t hv = float32_to_half(d.y);
      std::byte* px = row + static_cast<std::size_t>(x) * 4u;
      px[0] = static_cast<std::byte>(hu & 0xFFu);
      px[1] = static_cast<std::byte>((hu >> 8) & 0xFFu);
      px[2] = static_cast<std::byte>(hv & 0xFFu);
      px[3] = static_cast<std::byte>((hv >> 8) & 0xFFu);
    }
  }

  if (!flow_sink_.writeFrameWithFormat(flow_payload_.data(), static_cast<unsigned>(flow_pitch), f8::cppsdk::kVideoFormatFlow2F16)) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "flow shm write failed", "runtime", json::object());
    prev_gray_ = gray;
    return;
  }

  publish_state_if_changed("flowShmFormat", flow_shm_format_, "runtime", json::object());
  publish_state_if_changed("computeScale", compute_scale_, "runtime", json::object());
  publish_state_if_changed("lastError", "", "runtime", json::object());

  const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
  const std::uint64_t dense_vectors = static_cast<std::uint64_t>(std::max(0, flow.cols)) *
                                      static_cast<std::uint64_t>(std::max(0, flow.rows));
  emit_telemetry(end_ts_ms, hdr.frame_id, static_cast<double>(end_ts_ms - process_start_ms), dense_vectors);

  prev_gray_ = gray;
}

json DenseOptflowService::describe() {
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
                  "Compute flow once per N new frames.", false),
      state_field("flowShmName", schema_string(), "ro", "Flow SHM Name", "Output SHM name for UV flow field.", true),
      state_field("flowShmFormat", schema_string(), "ro", "Flow SHM Format", "Flow payload format. Fixed to flow2_f16.", false),
      state_field("computeScale", schema_number(0.125, 0.0625, 1.0), "rw", "Compute Scale",
                  "Farneback compute scale, flow is upscaled back to full size.", false),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message.", false),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array();
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array();
  service["dataOutPorts"] = json::array({
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
