#include "flow_metric_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/zenoh_naming.h"
#include "../common/service_runtime_utils.h"

namespace f8::cvkit::flow_metric {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_string_enum;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::video_frame_port;

namespace {

float half_to_float(std::uint16_t half_bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>((half_bits >> 15) & 0x1u);
  const std::uint32_t exp = static_cast<std::uint32_t>((half_bits >> 10) & 0x1Fu);
  const std::uint32_t frac = static_cast<std::uint32_t>(half_bits & 0x03FFu);
  std::uint32_t out_bits = 0;

  if (exp == 0) {
    if (frac == 0) {
      out_bits = sign << 31;
    } else {
      std::uint32_t mantissa = frac;
      int shift = 0;
      while ((mantissa & 0x0400u) == 0u) {
        mantissa <<= 1u;
        ++shift;
      }
      mantissa &= 0x03FFu;
      const std::uint32_t exp32 = static_cast<std::uint32_t>(127 - 15 - shift);
      out_bits = (sign << 31) | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exp == 0x1Fu) {
    out_bits = (sign << 31) | 0x7F800000u | (frac << 13);
  } else {
    const std::uint32_t exp32 = exp + (127 - 15);
    out_bits = (sign << 31) | (exp32 << 23) | (frac << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

}  // namespace

FlowMetricService::FlowMetricService(Config cfg) : cfg_(std::move(cfg)) {}

FlowMetricService::~FlowMetricService() { stop(); }

bool FlowMetricService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  bus_cfg.apply_runtime_backend(runtime_backend);
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Flow Metric";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  input_flow_stream_key_.clear();
  compute_every_n_frames_ = 1;
  metric_mode_ = MetricMode::Divergence;
  metric_mode_state_ = "divergence";
  metric_scale_ = 1.0;
  scalar_stream_key_ = f8::cppsdk::zenoh_data_key(cfg_.service_id, cfg_.service_id, "scalar");
  scalar_format_ = "scalar1_f32";

  input_zenoh_flow_.reset();
  flow_payload_.clear();
  last_frame_id_ = 0;
  last_flow_open_attempt_ms_ = 0;
  frame_counter_ = 0;
  scalar_payload_.clear();
  scalar_output_frame_id_ = 0;
  flow_u_.release();
  flow_v_.release();
  du_dx_.release();
  du_dy_.release();
  dv_dx_.release();
  dv_dy_.release();
  metric_output_.release();

  monitor_observed_frames_ = 0;
  monitor_processed_frames_ = 0;
  monitor_window_processed_frames_ = 0;
  monitor_fail_frames_ = 0;
  monitor_last_points_per_frame_ = 0;
  monitor_window_start_ms_ = 0;
  monitor_last_process_ms_ = 0.0;
  monitor_total_process_ms_ = 0.0;
  monitor_last_latency_ms_ = 0.0;
  monitor_total_latency_ms_ = 0.0;
  monitor_fps_ = 0.0;

  auto publisher = std::make_shared<f8::cppsdk::ZenohLatestVideoFramePublisher>();
  if (publisher->open(runtime_backend, scalar_stream_key_)) {
    scalar_zenoh_publisher_ = publisher;
    spdlog::info("flow_metric zenoh scalar publisher enabled serviceId={} key={}", cfg_.service_id, scalar_stream_key_);
  } else {
    scalar_zenoh_publisher_.reset();
    spdlog::error("flow_metric zenoh scalar publisher unavailable serviceId={} key={}", cfg_.service_id,
                  scalar_stream_key_);
    bus_->stop();
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("computeEveryNFrames", compute_every_n_frames_, "init", json::object());
  publish_state_if_changed("metricMode", metric_mode_state_, "init", json::object());
  publish_state_if_changed("metricScale", metric_scale_, "init", json::object());
  publish_state_if_changed("scalarFormat", scalar_format_, "init", json::object());
  publish_state_if_changed("scalarFrameSchemaVersion", 1, "init", json::object());
  publish_error_if_changed("", "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_flow_metric started serviceId={} backend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend));
  return true;
}

void FlowMetricService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();

  std::lock_guard<std::mutex> lock(io_mu_);
  if (input_zenoh_flow_) {
    input_zenoh_flow_->close();
  }
  input_zenoh_flow_.reset();
  if (scalar_zenoh_publisher_) {
    scalar_zenoh_publisher_->close();
  }
  scalar_zenoh_publisher_.reset();
}

void FlowMetricService::tick() {
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

void FlowMetricService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                                 const json& meta) {
  service_runtime::publish_state_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, field, value,
                                            source, meta);
}

void FlowMetricService::publish_error_if_changed(const json& value, const std::string& source, const json& meta) {
  service_runtime::publish_error_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, value, source,
                                            meta);
}

void FlowMetricService::emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms,
                                              double latency_ms, std::uint64_t points_per_frame) {
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
  monitor_last_points_per_frame_ = points_per_frame;

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
  metrics.last_points_per_frame = monitor_last_points_per_frame_;
  service_runtime::publish_cv_process_metrics(bus_.get(), metrics);
}

void FlowMetricService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

void FlowMetricService::on_state(const std::string& node_id, const std::string& field, const json& value, std::int64_t ts_ms,
                                 const json& meta) {
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

  if (field == "metricMode") {
    if (!value.is_string()) {
      publish_error_if_changed("invalid metricMode", "state", meta);
      return;
    }
    const std::string mode = service_runtime::to_lower_ascii_copy(service_runtime::trim_copy(value.get<std::string>()));
    if (mode == "divergence") {
      metric_mode_ = MetricMode::Divergence;
      metric_mode_state_ = "divergence";
    } else if (mode == "magnitude") {
      metric_mode_ = MetricMode::Magnitude;
      metric_mode_state_ = "magnitude";
    } else if (mode == "curl") {
      metric_mode_ = MetricMode::Curl;
      metric_mode_state_ = "curl";
    } else if (mode == "strain") {
      metric_mode_ = MetricMode::Strain;
      metric_mode_state_ = "strain";
    } else {
      publish_error_if_changed("invalid metricMode: " + value.get<std::string>(), "state", meta);
      return;
    }
    publish_state_if_changed("metricMode", metric_mode_state_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "metricScale") {
    double v = 0.0;
    if (!service_runtime::parse_json_double(value, v)) {
      publish_error_if_changed("invalid metricScale", "state", meta);
      return;
    }
    metric_scale_ = std::max(-1000.0, std::min(1000.0, v));
    publish_state_if_changed("metricScale", metric_scale_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
}

void FlowMetricService::on_data(const std::string& node_id, const std::string& port, const json& value, std::int64_t ts_ms,
                                const json& meta) {
  (void)node_id;
  (void)port;
  (void)value;
  (void)ts_ms;
  (void)meta;
  // Pull-based latest-frame stream input.
}

bool FlowMetricService::ensure_flow_open() {
  std::string key;
  if (bus_) {
    const auto resolved = bus_->data_input_zenoh_key(cfg_.service_id, "flow");
    if (resolved.has_value()) {
      key = service_runtime::trim_copy(*resolved);
    }
  }
  if (key.empty()) {
    publish_error_if_changed("missing flow data input", "runtime", json::object());
    return false;
  }
  if (input_zenoh_flow_ && input_zenoh_flow_->valid() && input_flow_stream_key_ == key) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_flow_open_attempt_ms_ > 0 && (now - last_flow_open_attempt_ms_) < 1000) {
    return false;
  }
  last_flow_open_attempt_ms_ = now;

  if (input_zenoh_flow_) {
    input_zenoh_flow_->close();
  }
  input_zenoh_flow_.reset();
  auto subscriber = std::make_unique<f8::cppsdk::ZenohLatestVideoFrameSubscriber>();
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  if (!subscriber->open(runtime_backend, key)) {
    publish_error_if_changed("zenoh flow subscribe failed: " + key, "runtime", json::object());
    return false;
  }
  input_flow_stream_key_ = key;
  input_zenoh_flow_ = std::move(subscriber);
  last_frame_id_ = 0;
  frame_counter_ = 0;
  flow_payload_.clear();
  flow_u_.release();
  flow_v_.release();
  du_dx_.release();
  du_dy_.release();
  dv_dx_.release();
  dv_dy_.release();
  metric_output_.release();
  publish_error_if_changed("", "runtime", json::object());
  return true;
}

void FlowMetricService::process_frame_once() {
  if (!bus_) return;

  std::lock_guard<std::mutex> lock(io_mu_);
  if (!ensure_flow_open()) {
    return;
  }

  if (!input_zenoh_flow_) {
    return;
  }
  auto latest = input_zenoh_flow_->wait_latest(std::chrono::milliseconds(20));
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
  flow_payload_ = std::move(latest->payload);

  ++monitor_observed_frames_;
  ++frame_counter_;
  last_frame_id_ = source_frame_id;

  const auto frame_validation =
      service_runtime::validate_frame_buffer(frame_format, frame_width, frame_height, frame_pitch, flow_payload_.size(),
                                             f8::cppsdk::kVideoFormatFlow2F16, 4u);
  if (!frame_validation.ok) {
    flow_u_.release();
    flow_v_.release();
    du_dx_.release();
    du_dy_.release();
    dv_dx_.release();
    dv_dy_.release();
    metric_output_.release();
    return;
  }
  const std::size_t row_bytes = frame_validation.row_bytes;

  if ((frame_counter_ % static_cast<std::uint64_t>(std::max(1, compute_every_n_frames_))) != 0) {
    return;
  }

  const std::int64_t process_start_ms = f8::cppsdk::now_ms();
  const int width = static_cast<int>(frame_width);
  const int height = static_cast<int>(frame_height);
  if (width <= 0 || height <= 0) {
    ++monitor_fail_frames_;
    publish_error_if_changed("invalid flow dimensions", "runtime", json::object());
    return;
  }

  flow_u_.create(height, width, CV_32FC1);
  flow_v_.create(height, width, CV_32FC1);
  for (int y = 0; y < height; ++y) {
    const std::byte* row = flow_payload_.data() + static_cast<std::size_t>(y) * row_bytes;
    float* row_u = flow_u_.ptr<float>(y);
    float* row_v = flow_v_.ptr<float>(y);
    for (int x = 0; x < width; ++x) {
      const std::byte* px = row + static_cast<std::size_t>(x) * 4u;
      const std::uint16_t half_u = static_cast<std::uint16_t>(static_cast<unsigned>(px[0])) |
                                   (static_cast<std::uint16_t>(static_cast<unsigned>(px[1])) << 8u);
      const std::uint16_t half_v = static_cast<std::uint16_t>(static_cast<unsigned>(px[2])) |
                                   (static_cast<std::uint16_t>(static_cast<unsigned>(px[3])) << 8u);
      row_u[x] = half_to_float(half_u);
      row_v[x] = half_to_float(half_v);
    }
  }

  try {
    switch (metric_mode_) {
      case MetricMode::Divergence:
        cv::Sobel(flow_u_, du_dx_, CV_32F, 1, 0, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(flow_v_, dv_dy_, CV_32F, 0, 1, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::add(du_dx_, dv_dy_, metric_output_);
        break;
      case MetricMode::Magnitude:
        cv::magnitude(flow_u_, flow_v_, metric_output_);
        break;
      case MetricMode::Curl:
        cv::Sobel(flow_v_, dv_dx_, CV_32F, 1, 0, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(flow_u_, du_dy_, CV_32F, 0, 1, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::subtract(dv_dx_, du_dy_, metric_output_);
        break;
      case MetricMode::Strain:
        cv::Sobel(flow_u_, du_dx_, CV_32F, 1, 0, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(flow_v_, dv_dy_, CV_32F, 0, 1, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(flow_u_, du_dy_, CV_32F, 0, 1, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(flow_v_, dv_dx_, CV_32F, 1, 0, 3, 0.5, 0.0, cv::BORDER_REPLICATE);
        cv::Mat extensional;
        cv::Mat shear;
        cv::subtract(du_dx_, dv_dy_, extensional);
        cv::add(du_dy_, dv_dx_, shear);
        cv::magnitude(extensional, shear, metric_output_);
        break;
    }
    metric_output_ *= static_cast<float>(metric_scale_);
  } catch (const cv::Exception& ex) {
    ++monitor_fail_frames_;
    publish_error_if_changed(std::string("opencv flow metric failed: ") + ex.what(), "runtime",
                             json::object());
    return;
  }

  auto pack_scalar_payload = [this, width, height](std::size_t scalar_pitch) {
    const std::size_t scalar_bytes = scalar_pitch * static_cast<std::size_t>(height);
    scalar_payload_.assign(scalar_bytes, std::byte{0});
    for (int y = 0; y < height; ++y) {
      const float* src = metric_output_.ptr<float>(y);
      std::byte* dst = scalar_payload_.data() + static_cast<std::size_t>(y) * scalar_pitch;
      std::memcpy(dst, src, static_cast<std::size_t>(width) * sizeof(float));
    }
  };

  if (!scalar_zenoh_publisher_ || !scalar_zenoh_publisher_->valid()) {
    ++monitor_fail_frames_;
    publish_error_if_changed("scalar zenoh publisher unavailable: " + scalar_stream_key_, "runtime", json::object());
    return;
  }
  const std::size_t scalar_pitch = static_cast<std::size_t>(width) * sizeof(float);
  pack_scalar_payload(scalar_pitch);

  f8::cppsdk::VideoFrameView frame;
  frame.width = static_cast<unsigned>(width);
  frame.height = static_cast<unsigned>(height);
  frame.pitch = static_cast<unsigned>(scalar_pitch);
  frame.format = f8::cppsdk::kVideoFormatScalar1F32;
  frame.frame_id = ++scalar_output_frame_id_;
  frame.ts_ms = f8::cppsdk::now_ms();
  frame.payload = scalar_payload_.data();
  frame.payload_bytes = scalar_payload_.size();
  if (!scalar_zenoh_publisher_->publish_frame(frame)) {
    ++monitor_fail_frames_;
    publish_error_if_changed("scalar zenoh publish failed: " + scalar_stream_key_, "runtime", json::object());
    return;
  }

  publish_state_if_changed("metricMode", metric_mode_state_, "runtime", json::object());
  publish_state_if_changed("metricScale", metric_scale_, "runtime", json::object());
  publish_state_if_changed("scalarFormat", scalar_format_, "runtime", json::object());
  publish_state_if_changed("scalarFrameSchemaVersion", 1, "runtime", json::object());
  publish_error_if_changed("", "runtime", json::object());

  const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
  const std::uint64_t points = static_cast<std::uint64_t>(std::max(0, width)) * static_cast<std::uint64_t>(std::max(0, height));
  emit_monitor_snapshot(end_ts_ms, source_frame_id, static_cast<double>(end_ts_ms - process_start_ms),
                        service_runtime::latency_ms_from_timestamps(end_ts_ms, latest->ts_ms), points);
}

json FlowMetricService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.flowmetric";
  service["label"] = "CVKit Flow Metric";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"cv", "optical_flow", "flow_metric", "scalar_field"});
  service["stateFields"] = json::array({
      state_field("computeEveryNFrames", schema_integer(1, 1, 120), "rw", "Compute Every N Frames",
                  "Compute selected flow metric once per N new flow frames.", false),
      state_field("metricMode", schema_string_enum({"divergence", "magnitude", "curl", "strain"}, "divergence"), "rw",
                  "Metric Mode", "Flow metric mode: divergence | magnitude | curl | strain.", false),
      state_field("metricScale", schema_number(1.0, -1000.0, 1000.0), "rw", "Metric Scale",
                  "Scale factor applied to computed metric values before output.", false),
      state_field("scalarFormat", schema_string_enum({"scalar1_f32"}), "ro", "Scalar Format",
                  "Output payload format. Fixed to scalar1_f32.", false),
      state_field("scalarFrameSchemaVersion", schema_integer(1, 1, 1), "ro", "Scalar Frame Schema",
                  "Output scalar frame schema version.", false),
  });
  service["commands"] = json::array();
  service["dataInPorts"] = json::array({
      video_frame_port("flow", "Input dense optical-flow frame stream."),
  });
  service["dataOutPorts"] = json::array({
      video_frame_port("scalar", "Scalar metric frame stream."),
  });

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::flow_metric
