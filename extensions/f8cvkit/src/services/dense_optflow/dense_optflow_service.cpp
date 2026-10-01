#include "dense_optflow_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/zenoh_naming.h"
#include "../common/service_runtime_utils.h"

namespace f8::cvkit::dense_optflow {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_object;
using f8::cppsdk::describe::schema_string_enum;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::video_frame_port;

namespace {

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
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  bus_cfg.apply_runtime_backend(runtime_backend);
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

  input_video_key_.clear();
  compute_every_n_frames_ = 2;
  flow_key_ = f8::cppsdk::zenoh_data_key(cfg_.service_id, cfg_.service_id, "flow");
  compute_scale_ = 0.5;

  input_zenoh_video_.reset();
  frame_bgra_.clear();
  flow_payload_.clear();
  flow_output_frame_id_ = 0;
  last_frame_id_ = 0;
  last_video_open_attempt_ms_ = 0;
  frame_counter_ = 0;

  prev_gray_.release();
  gray_.release();
  prev_compute_.release();
  gray_compute_.release();
  flow_compute_.release();
  has_prev_gray_ = false;
  prev_width_ = 0;
  prev_height_ = 0;

  monitor_observed_frames_ = 0;
  monitor_processed_frames_ = 0;
  monitor_window_processed_frames_ = 0;
  monitor_fail_frames_ = 0;
  monitor_last_vectors_per_frame_ = 0;
  monitor_window_start_ms_ = 0;
  monitor_last_process_ms_ = 0.0;
  monitor_total_process_ms_ = 0.0;
  monitor_last_latency_ms_ = 0.0;
  monitor_total_latency_ms_ = 0.0;
  monitor_fps_ = 0.0;

  auto publisher = std::make_shared<f8::cppsdk::ZenohLatestVideoFramePublisher>();
  if (publisher->open(runtime_backend, flow_key_)) {
    flow_zenoh_publisher_ = publisher;
    spdlog::info("dense_optflow zenoh flow publisher enabled serviceId={} key={}", cfg_.service_id, flow_key_);
  } else {
    flow_zenoh_publisher_.reset();
    spdlog::error("dense_optflow zenoh flow publisher unavailable serviceId={} key={}", cfg_.service_id, flow_key_);
    bus_->stop();
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("computeEveryNFrames", compute_every_n_frames_, "init", json::object());
  publish_state_if_changed("flowFormat", "flow2_f16", "init", json::object());
  publish_state_if_changed("flowFrameSchemaVersion", 1, "init", json::object());
  publish_state_if_changed("computeScale", compute_scale_, "init", json::object());
  publish_state_if_changed("flowOutputScaleX", 1.0, "init", json::object());
  publish_state_if_changed("flowOutputScaleY", 1.0, "init", json::object());
  publish_error_if_changed("", "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_dense_optflow started serviceId={} backend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend));
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
  if (input_zenoh_video_) {
    input_zenoh_video_->close();
  }
  input_zenoh_video_.reset();
  if (flow_zenoh_publisher_) {
    flow_zenoh_publisher_->close();
  }
  flow_zenoh_publisher_.reset();
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
  service_runtime::publish_state_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, field, value,
                                            source, meta);
}

void DenseOptflowService::publish_error_if_changed(const json& value, const std::string& source, const json& meta) {
  service_runtime::publish_error_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, value, source,
                                            meta);
}

void DenseOptflowService::emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms,
                                                double latency_ms, std::uint64_t vectors_per_frame) {
  if (!bus_) return;
  (void)frame_id;
  if (monitor_window_start_ms_ <= 0) {
    monitor_window_start_ms_ = ts_ms;
  }
  ++monitor_processed_frames_;
  ++monitor_window_processed_frames_;
  monitor_last_process_ms_ = process_ms;
  monitor_total_process_ms_ += process_ms;
  monitor_last_latency_ms_ = latency_ms;
  monitor_total_latency_ms_ += latency_ms;
  monitor_last_vectors_per_frame_ = vectors_per_frame;

  const std::int64_t elapsed = ts_ms - monitor_window_start_ms_;
  if (elapsed >= 1000) {
    monitor_fps_ = static_cast<double>(monitor_window_processed_frames_) * 1000.0 / static_cast<double>(elapsed);
    monitor_window_start_ms_ = ts_ms;
    monitor_window_processed_frames_ = 0;
  }

  const std::uint64_t dropped_frames = monitor_observed_frames_ > monitor_processed_frames_
                                           ? (monitor_observed_frames_ - monitor_processed_frames_)
                                           : 0;
  const double avg_process_ms = monitor_processed_frames_ > 0
                                    ? (monitor_total_process_ms_ / static_cast<double>(monitor_processed_frames_))
                                    : 0.0;
  const double avg_latency_ms = monitor_processed_frames_ > 0
                                    ? (monitor_total_latency_ms_ / static_cast<double>(monitor_processed_frames_))
                                    : 0.0;
  service_runtime::CvProcessMetrics metrics;
  metrics.observed_frames = monitor_observed_frames_;
  metrics.processed_frames = monitor_processed_frames_;
  metrics.dropped_frames = dropped_frames;
  metrics.failed_frames = monitor_fail_frames_;
  metrics.last_process_ms = monitor_last_process_ms_;
  metrics.avg_process_ms = avg_process_ms;
  metrics.last_latency_ms = monitor_last_latency_ms_;
  metrics.avg_latency_ms = avg_latency_ms;
  metrics.process_fps = monitor_fps_;
  metrics.last_vectors_per_frame = monitor_last_vectors_per_frame_;
  service_runtime::publish_cv_process_metrics(bus_.get(), metrics);
}

void DenseOptflowService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

void DenseOptflowService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                   std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;

  if (field == "computeEveryNFrames") {
    int v = 0;
    if (!service_runtime::parse_json_int(value, v)) {
      publish_error_if_changed("invalid computeEveryNFrames", "state", meta);
      return;
    }
    v = std::max(1, std::min(120, v));
    compute_every_n_frames_ = v;
    publish_state_if_changed("computeEveryNFrames", compute_every_n_frames_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "computeScale") {
    double v = 0.0;
    if (!service_runtime::parse_json_double(value, v)) {
      publish_error_if_changed("invalid computeScale", "state", meta);
      return;
    }
    compute_scale_ = std::max(0.25, std::min(1.0, v));
    publish_state_if_changed("computeScale", compute_scale_, "state", meta);
    publish_error_if_changed("", "state", meta);
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
  // Stream pull mode only.
}

bool DenseOptflowService::ensure_video_open() {
  std::string key;
  if (bus_) {
    const auto resolved = bus_->data_input_zenoh_key(cfg_.service_id, "video");
    if (resolved.has_value()) {
      key = service_runtime::trim_copy(*resolved);
    }
  }
  if (key.empty()) {
    publish_error_if_changed("missing video data input", "runtime", json::object());
    return false;
  }
  if (input_zenoh_video_ && input_zenoh_video_->valid() && input_video_key_ == key) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_video_open_attempt_ms_ > 0 && (now - last_video_open_attempt_ms_) < 1000) {
    return false;
  }
  last_video_open_attempt_ms_ = now;

  if (input_zenoh_video_) {
    input_zenoh_video_->close();
  }
  input_zenoh_video_.reset();
  auto subscriber = std::make_unique<f8::cppsdk::ZenohLatestVideoFrameSubscriber>();
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  if (!subscriber->open(runtime_backend, key)) {
    publish_error_if_changed("zenoh video subscribe failed: " + key, "runtime", json::object());
    return false;
  }
  input_video_key_ = key;
  input_zenoh_video_ = std::move(subscriber);
  last_frame_id_ = 0;
  frame_counter_ = 0;
  frame_bgra_.clear();
  prev_gray_.release();
  gray_.release();
  prev_compute_.release();
  gray_compute_.release();
  flow_compute_.release();
  has_prev_gray_ = false;
  prev_width_ = 0;
  prev_height_ = 0;
  publish_error_if_changed("", "runtime", json::object());
  return true;
}

void DenseOptflowService::process_frame_once() {
  if (!bus_) return;

  std::lock_guard<std::mutex> lock(flow_mu_);
  if (!ensure_video_open()) {
    return;
  }

  if (!input_zenoh_video_) {
    return;
  }
  auto latest = input_zenoh_video_->wait_latest(std::chrono::milliseconds(20));
  if (!latest.has_value()) {
    return;
  }
  if (latest->frame_id == 0 || latest->frame_id == last_frame_id_) {
    return;
  }
  const unsigned frame_width = latest->width;
  const unsigned frame_height = latest->height;
  const unsigned frame_pitch = latest->pitch;
  const std::uint32_t frame_format = latest->format;
  const std::uint64_t source_frame_id = latest->frame_id;
  frame_bgra_ = std::move(latest->payload);

  ++monitor_observed_frames_;
  ++frame_counter_;

  const int every_n = std::max(1, compute_every_n_frames_);
  if (has_prev_gray_ && (frame_counter_ % static_cast<std::uint64_t>(every_n)) != 0) {
    last_frame_id_ = source_frame_id;
    return;
  }

  last_frame_id_ = source_frame_id;

  const auto frame_validation = service_runtime::validate_frame_buffer(
      frame_format, frame_width, frame_height, frame_pitch, frame_bgra_.size(), f8::cppsdk::kVideoFormatBgra32, 4u);
  if (!frame_validation.ok) {
    has_prev_gray_ = false;
    prev_gray_.release();
    prev_compute_.release();
    gray_compute_.release();
    flow_compute_.release();
    return;
  }
  const std::size_t row_bytes = frame_validation.row_bytes;

  cv::Mat bgra(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC4,
               const_cast<std::byte*>(frame_bgra_.data()), row_bytes);
  try {
    cv::cvtColor(bgra, gray_, cv::COLOR_BGRA2GRAY);
  } catch (const cv::Exception& ex) {
    ++monitor_fail_frames_;
    publish_error_if_changed(std::string("opencv cvtColor failed: ") + ex.what(), "runtime", json::object());
    return;
  }

  if (!has_prev_gray_ || prev_width_ != static_cast<int>(frame_width) || prev_height_ != static_cast<int>(frame_height)) {
    gray_.copyTo(prev_gray_);
    has_prev_gray_ = true;
    prev_width_ = static_cast<int>(frame_width);
    prev_height_ = static_cast<int>(frame_height);
    return;
  }

  const std::int64_t process_start_ms = f8::cppsdk::now_ms();

  const double scale = std::max(0.25, std::min(1.0, compute_scale_));
  compute_scale_ = scale;

  cv::Mat prev_compute = prev_gray_;
  cv::Mat gray_compute = gray_;
  if (scale < 0.999) {
    int sw = static_cast<int>(std::lround(static_cast<double>(gray_.cols) * scale));
    int sh = static_cast<int>(std::lround(static_cast<double>(gray_.rows) * scale));
    sw = std::max(sw, 1);
    sh = std::max(sh, 1);
    try {
      cv::resize(prev_gray_, prev_compute_, cv::Size(sw, sh), 0.0, 0.0, cv::INTER_AREA);
      cv::resize(gray_, gray_compute_, cv::Size(sw, sh), 0.0, 0.0, cv::INTER_AREA);
      prev_compute = prev_compute_;
      gray_compute = gray_compute_;
    } catch (const cv::Exception& ex) {
      ++monitor_fail_frames_;
      publish_error_if_changed(std::string("opencv resize failed: ") + ex.what(), "runtime", json::object());
      gray_.copyTo(prev_gray_);
      return;
    }
  }

  try {
    cv::calcOpticalFlowFarneback(prev_compute, gray_compute, flow_compute_, 0.5, 3, 15, 3, 5, 1.2, 0);
  } catch (const cv::Exception& ex) {
    ++monitor_fail_frames_;
    publish_error_if_changed(std::string("opencv farneback failed: ") + ex.what(), "runtime", json::object());
    gray_.copyTo(prev_gray_);
    return;
  }

  cv::Mat flow = flow_compute_;

  const bool publish_zenoh_flow = flow_zenoh_publisher_ && flow_zenoh_publisher_->valid();

  auto pack_flow_payload = [this, &flow](std::size_t flow_pitch) {
    const std::size_t flow_bytes = flow_pitch * static_cast<std::size_t>(flow.rows);
    if (flow_payload_.size() != flow_bytes) {
      flow_payload_.assign(flow_bytes, std::byte{0});
    }
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
  };

  if (!publish_zenoh_flow) {
    ++monitor_fail_frames_;
    publish_error_if_changed("flow zenoh publisher unavailable: " + flow_key_, "runtime", json::object());
    gray_.copyTo(prev_gray_);
    return;
  }
  const std::size_t flow_pitch = static_cast<std::size_t>(flow.cols) * 4u;
  pack_flow_payload(flow_pitch);

  f8::cppsdk::VideoFrameView frame;
  frame.width = static_cast<unsigned>(flow.cols);
  frame.height = static_cast<unsigned>(flow.rows);
  frame.pitch = static_cast<unsigned>(flow_pitch);
  frame.format = f8::cppsdk::kVideoFormatFlow2F16;
  frame.frame_id = ++flow_output_frame_id_;
  frame.ts_ms = f8::cppsdk::now_ms();
  frame.payload = flow_payload_.data();
  frame.payload_bytes = flow_payload_.size();
  if (!flow_zenoh_publisher_->publish_frame(frame)) {
    ++monitor_fail_frames_;
    publish_error_if_changed("flow zenoh publish failed: " + flow_key_, "runtime", json::object());
    gray_.copyTo(prev_gray_);
    return;
  }

  publish_state_if_changed("flowFormat", "flow2_f16", "runtime", json::object());
  publish_state_if_changed("flowFrameSchemaVersion", 1, "runtime", json::object());
  publish_state_if_changed("computeScale", compute_scale_, "runtime", json::object());
  publish_state_if_changed("flowOutputScaleX", static_cast<double>(flow.cols) / static_cast<double>(std::max(1, gray_.cols)),
                           "runtime", json::object());
  publish_state_if_changed("flowOutputScaleY", static_cast<double>(flow.rows) / static_cast<double>(std::max(1, gray_.rows)),
                           "runtime", json::object());
  publish_error_if_changed("", "runtime", json::object());

  const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
  const std::uint64_t dense_vectors = static_cast<std::uint64_t>(std::max(0, flow.cols)) *
                                      static_cast<std::uint64_t>(std::max(0, flow.rows));
  emit_monitor_snapshot(end_ts_ms, source_frame_id, static_cast<double>(end_ts_ms - process_start_ms),
                        service_runtime::latency_ms_from_timestamps(end_ts_ms, latest->ts_ms), dense_vectors);

  gray_.copyTo(prev_gray_);
}

json DenseOptflowService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.denseoptflow";
  service["label"] = "CVKit Dense Optical Flow";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"cv", "optical_flow", "flow_field"});
  service["stateFields"] = json::array({
      state_field("computeEveryNFrames", schema_integer(2, 1, 120), "rw", "Compute Every N Frames",
                  "Compute flow once per N new frames.", false),
      state_field("flowFormat", schema_string_enum({"flow2_f16"}), "ro", "Flow Format",
                  "Flow payload format. Fixed to flow2_f16.", false),
      state_field("flowFrameSchemaVersion", schema_integer(1, 1, 1), "ro", "Flow Frame Schema",
                  "Output flow frame schema version.", false),
      state_field("computeScale", schema_number(0.5, 0.25, 1.0), "rw", "Compute Scale",
                  "Farneback compute scale; output flow stays at compute scale.", false),
      state_field("flowOutputScaleX", schema_number(), "ro", "Flow Output Scale X", "Output flow width / source width.", false),
      state_field("flowOutputScaleY", schema_number(), "ro", "Flow Output Scale Y", "Output flow height / source height.", false),
  });
  service["commands"] = json::array();
  service["dataInPorts"] = json::array({
      video_frame_port("video", "Input video frame stream."),
  });
  service["dataOutPorts"] = json::array({
      video_frame_port("flow", "Dense optical-flow frame stream."),
  });

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::dense_optflow
