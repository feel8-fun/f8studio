#include "video_stab_service.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/zenoh_naming.h"
#include "../common/service_runtime_utils.h"

namespace f8::cvkit::video_stab {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_boolean;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_object;
using f8::cppsdk::describe::schema_string;
using f8::cppsdk::describe::schema_string_enum;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::video_frame_port;

namespace {

int count_inliers(const cv::Mat& inlier_mask) {
  if (inlier_mask.empty())
    return 0;
  if (inlier_mask.type() != CV_8UC1)
    return 0;
  return cv::countNonZero(inlier_mask);
}

VideoStabService::MotionParams motion_from_affine(const cv::Mat& affine_2x3) {
  VideoStabService::MotionParams out;
  if (affine_2x3.empty() || affine_2x3.rows != 2 || affine_2x3.cols != 3)
    return out;
  const double a = affine_2x3.at<double>(0, 0);
  const double b = affine_2x3.at<double>(0, 1);
  const double c = affine_2x3.at<double>(1, 0);
  out.tx = affine_2x3.at<double>(0, 2);
  out.ty = affine_2x3.at<double>(1, 2);
  out.angle_deg = std::atan2(c, a) * 180.0 / CV_PI;
  const double scale = std::sqrt(a * a + c * c);
  out.scale = (scale > 1e-6) ? scale : 1.0;
  return out;
}

VideoStabService::MotionParams motion_from_homography(const cv::Mat& homography_3x3) {
  VideoStabService::MotionParams out;
  if (homography_3x3.empty() || homography_3x3.rows != 3 || homography_3x3.cols != 3)
    return out;
  const double h00 = homography_3x3.at<double>(0, 0);
  const double h10 = homography_3x3.at<double>(1, 0);
  out.tx = homography_3x3.at<double>(0, 2);
  out.ty = homography_3x3.at<double>(1, 2);
  out.angle_deg = std::atan2(h10, h00) * 180.0 / CV_PI;
  const double scale = std::sqrt(h00 * h00 + h10 * h10);
  out.scale = (scale > 1e-6) ? scale : 1.0;
  return out;
}

VideoStabService::MotionParams lerp_motion(const VideoStabService::MotionParams& prev,
                                           const VideoStabService::MotionParams& curr, double alpha) {
  VideoStabService::MotionParams out;
  out.tx = alpha * curr.tx + (1.0 - alpha) * prev.tx;
  out.ty = alpha * curr.ty + (1.0 - alpha) * prev.ty;
  out.angle_deg = alpha * curr.angle_deg + (1.0 - alpha) * prev.angle_deg;
  out.scale = alpha * curr.scale + (1.0 - alpha) * prev.scale;
  if (out.scale <= 1e-6)
    out.scale = 1.0;
  return out;
}

cv::Mat correction_affine_2x3(const VideoStabService::MotionParams& raw, const VideoStabService::MotionParams& smooth,
                              int width, int height) {
  const double corr_tx = smooth.tx - raw.tx;
  const double corr_ty = smooth.ty - raw.ty;
  const double corr_angle_deg = smooth.angle_deg - raw.angle_deg;
  const double raw_scale = std::max(raw.scale, 1e-6);
  const double corr_scale = std::max(0.1, std::min(10.0, smooth.scale / raw_scale));

  const double rad = corr_angle_deg * CV_PI / 180.0;
  const double c = std::cos(rad) * corr_scale;
  const double s = std::sin(rad) * corr_scale;
  const double cx = static_cast<double>(width) * 0.5;
  const double cy = static_cast<double>(height) * 0.5;

  cv::Mat m = cv::Mat::eye(2, 3, CV_64F);
  m.at<double>(0, 0) = c;
  m.at<double>(0, 1) = -s;
  m.at<double>(1, 0) = s;
  m.at<double>(1, 1) = c;
  m.at<double>(0, 2) = corr_tx + cx - (c * cx - s * cy);
  m.at<double>(1, 2) = corr_ty + cy - (s * cx + c * cy);
  return m;
}

cv::Mat affine_2x3_to_homography_3x3(const cv::Mat& affine_2x3) {
  cv::Mat h = cv::Mat::eye(3, 3, CV_64F);
  h.at<double>(0, 0) = affine_2x3.at<double>(0, 0);
  h.at<double>(0, 1) = affine_2x3.at<double>(0, 1);
  h.at<double>(0, 2) = affine_2x3.at<double>(0, 2);
  h.at<double>(1, 0) = affine_2x3.at<double>(1, 0);
  h.at<double>(1, 1) = affine_2x3.at<double>(1, 1);
  h.at<double>(1, 2) = affine_2x3.at<double>(1, 2);
  return h;
}

}  // namespace

VideoStabService::VideoStabService(Config cfg) : cfg_(std::move(cfg)) {}

VideoStabService::~VideoStabService() {
  stop();
}

bool VideoStabService::start() {
  if (running_.load(std::memory_order_acquire))
    return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  bus_cfg.apply_runtime_backend(runtime_backend);
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Video Stabilizer";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);
  bus_->add_command_node(this, VideoStabService::describe());

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  input_stream_key_.clear();
  input_zenoh_video_.reset();
  output_stream_key_ = f8::cppsdk::zenoh_data_key(cfg_.service_id, cfg_.service_id, "video");
  output_frame_id_ = 0;
  output_initialized_ = false;
  has_prev_gray_ = false;
  prev_frame_width_ = 0;
  prev_frame_height_ = 0;
  prev_gray_.release();
  smooth_initialized_ = false;
  smooth_params_ = MotionParams{};
  trajectory_initialized_ = false;
  trajectory_raw_params_ = MotionParams{};
  trajectory_smooth_params_ = MotionParams{};
  consecutive_failures_ = 0;
  scene_change_count_ = 0;
  scene_cut_cooldown_remaining_ = 0;

  input_last_frame_id_ = 0;
  input_last_open_attempt_ms_ = 0;

  monitor_observed_frames_ = 0;
  monitor_processed_frames_ = 0;
  monitor_window_processed_frames_ = 0;
  monitor_fail_frames_ = 0;
  monitor_window_start_ms_ = 0;
  monitor_last_process_ms_ = 0.0;
  monitor_total_process_ms_ = 0.0;
  monitor_last_latency_ms_ = 0.0;
  monitor_total_latency_ms_ = 0.0;
  monitor_fps_ = 0.0;

  auto publisher = std::make_shared<f8::cppsdk::ZenohLatestVideoFramePublisher>();
  if (publisher->open(runtime_backend, output_stream_key_)) {
    output_zenoh_video_ = publisher;
    spdlog::info("video_stab zenoh video publisher enabled serviceId={} key={}", cfg_.service_id, output_stream_key_);
  } else {
    output_zenoh_video_.reset();
    spdlog::error("video_stab zenoh video publisher unavailable serviceId={} key={}", cfg_.service_id, output_stream_key_);
    bus_->stop();
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("videoFormat", "bgra32", "init", json::object());
  publish_state_if_changed("videoFrameSchemaVersion", 1, "init", json::object());
  publish_state_if_changed("motionModel", motion_model_state_, "init", json::object());
  publish_state_if_changed("stabilizationMode", stabilization_mode_state_, "init", json::object());
  publish_state_if_changed("smoothAlpha", smooth_alpha_, "init", json::object());
  publish_state_if_changed("maxCornerCount", max_corner_count_, "init", json::object());
  publish_state_if_changed("qualityLevel", quality_level_, "init", json::object());
  publish_state_if_changed("minDistance", min_distance_, "init", json::object());
  publish_state_if_changed("ransacReprojThreshold", ransac_reproj_threshold_, "init", json::object());
  publish_state_if_changed("resetOnFailureFrames", reset_on_failure_frames_, "init", json::object());
  publish_state_if_changed("sceneCutEnabled", scene_cut_enabled_, "init", json::object());
  publish_state_if_changed("sceneCutFrameDiffThreshold", scene_cut_frame_diff_threshold_, "init", json::object());
  publish_state_if_changed("sceneCutTrackRatioThreshold", scene_cut_track_ratio_threshold_, "init", json::object());
  publish_state_if_changed("sceneCutCooldownFrames", scene_cut_cooldown_frames_, "init", json::object());
  publish_error_if_changed("", "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_video_stab started serviceId={} backend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend));
  return true;
}

void VideoStabService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();

  std::lock_guard<std::mutex> lock(io_mu_);
  if (input_zenoh_video_) {
    input_zenoh_video_->close();
  }
  input_zenoh_video_.reset();
  if (output_zenoh_video_) {
    output_zenoh_video_->close();
  }
  output_zenoh_video_.reset();
  output_initialized_ = false;
}

void VideoStabService::tick() {
  if (!running())
    return;
  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
  }
  if (!active_.load(std::memory_order_acquire))
    return;
  process_frame_once();
}

void VideoStabService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                                const json& meta) {
  service_runtime::publish_state_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, field, value,
                                            source, meta);
}

void VideoStabService::publish_error_if_changed(const json& value, const std::string& source, const json& meta) {
  service_runtime::publish_error_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, value, source,
                                            meta);
}

void VideoStabService::emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms,
                                             double latency_ms) {
  if (!bus_)
    return;
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

  const std::int64_t elapsed = ts_ms - monitor_window_start_ms_;
  if (elapsed >= 1000) {
    monitor_fps_ = static_cast<double>(monitor_window_processed_frames_) * 1000.0 / static_cast<double>(elapsed);
    monitor_window_start_ms_ = ts_ms;
    monitor_window_processed_frames_ = 0;
  }

  const std::uint64_t dropped_frames =
      monitor_observed_frames_ > monitor_processed_frames_ ? (monitor_observed_frames_ - monitor_processed_frames_) : 0;
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
  service_runtime::publish_cv_process_metrics(bus_.get(), metrics);
}

void VideoStabService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

bool VideoStabService::parse_double_field(const json& value, double& out) const {
  return service_runtime::parse_json_double(value, out);
}

bool VideoStabService::parse_int_field(const json& value, int& out) const {
  return service_runtime::parse_json_int(value, out);
}

void VideoStabService::set_motion_model(const std::string& model, const json& meta) {
  const std::string normalized = service_runtime::to_lower_ascii_copy(service_runtime::trim_copy(model));
  if (normalized == "affine") {
    motion_model_ = MotionModel::Affine;
    motion_model_state_ = "affine";
  } else if (normalized == "homography") {
    motion_model_ = MotionModel::Homography;
    motion_model_state_ = "homography";
  } else {
    publish_error_if_changed("invalid motionModel: " + model, "state", meta);
    return;
  }
  reset_stabilizer_internal(meta, "motion_model_changed");
  publish_state_if_changed("motionModel", motion_model_state_, "state", meta);
  publish_error_if_changed("", "state", meta);
}

void VideoStabService::set_stabilization_mode(const std::string& mode, const json& meta) {
  const std::string normalized = service_runtime::to_lower_ascii_copy(service_runtime::trim_copy(mode));
  if (normalized == "trajectory") {
    stabilization_mode_ = StabilizationMode::Trajectory;
    stabilization_mode_state_ = "trajectory";
  } else if (normalized == "instant") {
    stabilization_mode_ = StabilizationMode::Instant;
    stabilization_mode_state_ = "instant";
  } else {
    publish_error_if_changed("invalid stabilizationMode: " + mode, "state", meta);
    return;
  }
  reset_stabilizer_internal(meta, "stabilization_mode_changed");
  publish_state_if_changed("stabilizationMode", stabilization_mode_state_, "state", meta);
  publish_error_if_changed("", "state", meta);
}

void VideoStabService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id)
    return;

  if (field == "motionModel" && value.is_string()) {
    set_motion_model(value.get<std::string>(), meta);
    return;
  }
  if (field == "stabilizationMode" && value.is_string()) {
    set_stabilization_mode(value.get<std::string>(), meta);
    return;
  }

  if (field == "smoothAlpha") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_error_if_changed("invalid smoothAlpha", "state", meta);
      return;
    }
    smooth_alpha_ = std::max(0.01, std::min(0.5, v));
    publish_state_if_changed("smoothAlpha", smooth_alpha_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "maxCornerCount") {
    int v = 0;
    if (!parse_int_field(value, v)) {
      publish_error_if_changed("invalid maxCornerCount", "state", meta);
      return;
    }
    max_corner_count_ = std::max(20, std::min(2000, v));
    publish_state_if_changed("maxCornerCount", max_corner_count_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "qualityLevel") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_error_if_changed("invalid qualityLevel", "state", meta);
      return;
    }
    quality_level_ = std::max(0.0001, std::min(0.3, v));
    publish_state_if_changed("qualityLevel", quality_level_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "minDistance") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_error_if_changed("invalid minDistance", "state", meta);
      return;
    }
    min_distance_ = std::max(1.0, std::min(100.0, v));
    publish_state_if_changed("minDistance", min_distance_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "ransacReprojThreshold") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_error_if_changed("invalid ransacReprojThreshold", "state", meta);
      return;
    }
    ransac_reproj_threshold_ = std::max(0.1, std::min(20.0, v));
    publish_state_if_changed("ransacReprojThreshold", ransac_reproj_threshold_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "resetOnFailureFrames") {
    int v = 0;
    if (!parse_int_field(value, v)) {
      publish_error_if_changed("invalid resetOnFailureFrames", "state", meta);
      return;
    }
    reset_on_failure_frames_ = std::max(1, std::min(120, v));
    publish_state_if_changed("resetOnFailureFrames", reset_on_failure_frames_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "sceneCutEnabled") {
    if (!value.is_boolean()) {
      publish_error_if_changed("invalid sceneCutEnabled", "state", meta);
      return;
    }
    scene_cut_enabled_ = value.get<bool>();
    publish_state_if_changed("sceneCutEnabled", scene_cut_enabled_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "sceneCutFrameDiffThreshold") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_error_if_changed("invalid sceneCutFrameDiffThreshold", "state", meta);
      return;
    }
    scene_cut_frame_diff_threshold_ = std::max(1.0, std::min(80.0, v));
    publish_state_if_changed("sceneCutFrameDiffThreshold", scene_cut_frame_diff_threshold_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "sceneCutTrackRatioThreshold") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_error_if_changed("invalid sceneCutTrackRatioThreshold", "state", meta);
      return;
    }
    scene_cut_track_ratio_threshold_ = std::max(0.01, std::min(0.95, v));
    publish_state_if_changed("sceneCutTrackRatioThreshold", scene_cut_track_ratio_threshold_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }

  if (field == "sceneCutCooldownFrames") {
    int v = 0;
    if (!parse_int_field(value, v)) {
      publish_error_if_changed("invalid sceneCutCooldownFrames", "state", meta);
      return;
    }
    scene_cut_cooldown_frames_ = std::max(0, std::min(120, v));
    publish_state_if_changed("sceneCutCooldownFrames", scene_cut_cooldown_frames_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
}

void VideoStabService::on_data(const std::string& node_id, const std::string& port, const json& value,
                               std::int64_t ts_ms, const json& meta) {
  (void)node_id;
  (void)port;
  (void)value;
  (void)ts_ms;
  (void)meta;
  // Pull-based latest-frame stream input.
}

bool VideoStabService::on_command(const std::string& call, const json& args, const json& meta, json& result,
                                  std::string& error_code, std::string& error_message) {
  (void)args;
  error_code.clear();
  error_message.clear();
  result = json::object();

  if (call == "resetStabilizer") {
    reset_stabilizer_internal(meta, "command");
    result["reset"] = true;
    return true;
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void VideoStabService::reset_stabilizer_internal(const json& meta, const std::string& reason) {
  (void)meta;
  (void)reason;
  has_prev_gray_ = false;
  prev_frame_width_ = 0;
  prev_frame_height_ = 0;
  prev_gray_.release();
  smooth_initialized_ = false;
  smooth_params_ = MotionParams{};
  trajectory_initialized_ = false;
  trajectory_raw_params_ = MotionParams{};
  trajectory_smooth_params_ = MotionParams{};
  consecutive_failures_ = 0;
  scene_cut_cooldown_remaining_ = 0;
}

bool VideoStabService::ensure_input_open() {
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
  if (input_zenoh_video_ && input_zenoh_video_->valid() && input_stream_key_ == key) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (input_last_open_attempt_ms_ > 0 && (now - input_last_open_attempt_ms_) < 1000) {
    return false;
  }
  input_last_open_attempt_ms_ = now;

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
  input_stream_key_ = key;
  input_zenoh_video_ = std::move(subscriber);
  input_last_frame_id_ = 0;
  input_frame_bgra_.clear();
  reset_stabilizer_internal(json::object(), "input_stream_changed");
  publish_error_if_changed("", "runtime", json::object());
  return true;
}

bool VideoStabService::ensure_output_open() {
  if (output_initialized_)
    return true;

  if (output_zenoh_video_ && output_zenoh_video_->valid()) {
    output_initialized_ = true;
    publish_state_if_changed("videoFormat", "bgra32", "runtime", json::object());
    publish_state_if_changed("videoFrameSchemaVersion", 1, "runtime", json::object());
    publish_error_if_changed("", "runtime", json::object());
    return true;
  }

  publish_error_if_changed("output zenoh publisher unavailable: " + output_stream_key_, "runtime", json::object());
  return false;
}

void VideoStabService::process_frame_once() {
  if (!bus_)
    return;

  std::lock_guard<std::mutex> lock(io_mu_);

  if (!ensure_input_open()) {
    return;
  }
  if (!ensure_output_open()) {
    return;
  }

  if (!input_zenoh_video_) {
    return;
  }
  auto latest = input_zenoh_video_->wait_latest(std::chrono::milliseconds(20));
  if (!latest.has_value()) {
    return;
  }
  if (latest->frame_id == 0 || latest->frame_id == input_last_frame_id_) {
    return;
  }
  const unsigned frame_width = latest->width;
  const unsigned frame_height = latest->height;
  const unsigned frame_pitch = latest->pitch;
  const std::uint32_t frame_format = latest->format;
  const std::uint64_t source_frame_id = latest->frame_id;
  const std::int64_t source_ts_ms = latest->ts_ms;
  input_frame_bgra_ = std::move(latest->payload);

  ++monitor_observed_frames_;
  input_last_frame_id_ = source_frame_id;
  const std::int64_t process_start_ms = f8::cppsdk::now_ms();

  const auto frame_validation = service_runtime::validate_frame_buffer(
      frame_format, frame_width, frame_height, frame_pitch, input_frame_bgra_.size(), f8::cppsdk::kVideoFormatBgra32, 4u);
  if (!frame_validation.ok) {
    reset_stabilizer_internal(json::object(), "transient_invalid_frame");
    return;
  }
  const std::size_t row_bytes = frame_validation.row_bytes;

  cv::Mat src_bgra(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC4,
                   const_cast<std::byte*>(input_frame_bgra_.data()), row_bytes);

  cv::Mat gray;
  try {
    cv::cvtColor(src_bgra, gray, cv::COLOR_BGRA2GRAY);
  } catch (const cv::Exception& ex) {
    ++monitor_fail_frames_;
    publish_error_if_changed(std::string("opencv cvtColor failed: ") + ex.what(), "runtime",
                             json::object());
    return;
  }

  cv::Mat stabilized = src_bgra.clone();
  bool motion_valid = false;
  bool scene_changed = false;
  int inlier_count = 0;
  int tracked_points = 0;
  int prev_points_count = 0;
  double scene_cut_frame_diff = 0.0;
  double scene_cut_track_ratio = 1.0;
  MotionParams raw_params{};
  MotionParams correction_raw_params{};
  MotionParams correction_smooth_params{};
  MotionParams corr_params{};

  if (!has_prev_gray_) {
    prev_gray_ = gray;
    has_prev_gray_ = true;
    prev_frame_width_ = static_cast<int>(frame_width);
    prev_frame_height_ = static_cast<int>(frame_height);
  } else if (prev_frame_width_ != static_cast<int>(frame_width) ||
             prev_frame_height_ != static_cast<int>(frame_height)) {
    reset_stabilizer_internal(json::object(), "video_dimensions_changed");
    prev_gray_ = gray;
    has_prev_gray_ = true;
    prev_frame_width_ = static_cast<int>(frame_width);
    prev_frame_height_ = static_cast<int>(frame_height);
  } else {
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> curr_pts;
    std::vector<unsigned char> status;
    std::vector<float> err;

    try {
      cv::goodFeaturesToTrack(prev_gray_, prev_pts, max_corner_count_, quality_level_, min_distance_);
      prev_points_count = static_cast<int>(prev_pts.size());
      if (prev_pts.size() >= 8) {
        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts, curr_pts, status, err);
      }
    } catch (const cv::Exception& ex) {
      ++monitor_fail_frames_;
      ++consecutive_failures_;
      publish_error_if_changed(std::string("opencv optical flow failed: ") + ex.what(), "runtime",
                               json::object());
      if (consecutive_failures_ >= reset_on_failure_frames_) {
        reset_stabilizer_internal(json::object(), "opencv_exception");
      }
      prev_gray_ = gray;
      return;
    }

    std::vector<cv::Point2f> prev_valid;
    std::vector<cv::Point2f> curr_valid;
    prev_valid.reserve(prev_pts.size());
    curr_valid.reserve(curr_pts.size());
    const int w = static_cast<int>(frame_width);
    const int h = static_cast<int>(frame_height);

    for (std::size_t i = 0; i < prev_pts.size() && i < curr_pts.size() && i < status.size(); ++i) {
      if (status[i] == 0)
        continue;
      const cv::Point2f p0 = prev_pts[i];
      const cv::Point2f p1 = curr_pts[i];
      if (p0.x < 0.0f || p0.y < 0.0f || p0.x >= static_cast<float>(w) || p0.y >= static_cast<float>(h))
        continue;
      if (p1.x < 0.0f || p1.y < 0.0f || p1.x >= static_cast<float>(w) || p1.y >= static_cast<float>(h))
        continue;
      prev_valid.push_back(p0);
      curr_valid.push_back(p1);
    }

    tracked_points = static_cast<int>(curr_valid.size());
    cv::Mat gray_diff;
    cv::absdiff(prev_gray_, gray, gray_diff);
    scene_cut_frame_diff = cv::mean(gray_diff)[0];
    const int track_ratio_denominator = std::max(prev_points_count, 1);
    scene_cut_track_ratio = static_cast<double>(tracked_points) / static_cast<double>(track_ratio_denominator);

    if (scene_cut_cooldown_remaining_ > 0) {
      --scene_cut_cooldown_remaining_;
    }

    const bool scene_cut_by_track_drop =
        scene_cut_frame_diff >= scene_cut_frame_diff_threshold_ &&
        (scene_cut_track_ratio <= scene_cut_track_ratio_threshold_ || tracked_points < 8);
    const bool scene_cut_by_hard_diff = scene_cut_frame_diff >= (scene_cut_frame_diff_threshold_ * 1.8);
    const bool scene_cut_triggered =
        scene_cut_enabled_ && scene_cut_cooldown_remaining_ <= 0 && (scene_cut_by_track_drop || scene_cut_by_hard_diff);

    if (scene_cut_triggered) {
      scene_changed = true;
      ++scene_change_count_;
      reset_stabilizer_internal(json::object(), "scene_cut");
      scene_cut_cooldown_remaining_ = scene_cut_cooldown_frames_;
      prev_gray_ = gray;
      has_prev_gray_ = true;
      prev_frame_width_ = static_cast<int>(frame_width);
      prev_frame_height_ = static_cast<int>(frame_height);
      consecutive_failures_ = 0;
      publish_error_if_changed("", "runtime", json::object());
    } else if (tracked_points >= 8) {
      cv::Mat inliers;
      try {
        if (motion_model_ == MotionModel::Affine) {
          cv::Mat affine =
              cv::estimateAffinePartial2D(prev_valid, curr_valid, inliers, cv::RANSAC, ransac_reproj_threshold_);
          if (!affine.empty() && affine.rows == 2 && affine.cols == 3) {
            raw_params = motion_from_affine(affine);
            inlier_count = count_inliers(inliers);
            motion_valid = true;
          }
        } else {
          cv::Mat homography =
              cv::findHomography(prev_valid, curr_valid, cv::RANSAC, ransac_reproj_threshold_, inliers);
          if (!homography.empty() && homography.rows == 3 && homography.cols == 3) {
            raw_params = motion_from_homography(homography);
            inlier_count = count_inliers(inliers);
            motion_valid = true;
          }
        }
      } catch (const cv::Exception& ex) {
        ++monitor_fail_frames_;
        ++consecutive_failures_;
        publish_error_if_changed(std::string("opencv transform estimate failed: ") + ex.what(), "runtime",
                                 json::object());
      }
    }

    if (motion_valid) {
      if (stabilization_mode_ == StabilizationMode::Trajectory) {
        if (!trajectory_initialized_) {
          trajectory_raw_params_ = raw_params;
          trajectory_smooth_params_ = raw_params;
          trajectory_initialized_ = true;
        } else {
          trajectory_raw_params_.tx += raw_params.tx;
          trajectory_raw_params_.ty += raw_params.ty;
          trajectory_raw_params_.angle_deg += raw_params.angle_deg;
          trajectory_raw_params_.scale *= raw_params.scale;
          trajectory_smooth_params_ = lerp_motion(trajectory_smooth_params_, trajectory_raw_params_, smooth_alpha_);
        }
        correction_raw_params = trajectory_raw_params_;
        correction_smooth_params = trajectory_smooth_params_;
      } else {
        if (!smooth_initialized_) {
          smooth_params_ = raw_params;
          smooth_initialized_ = true;
        } else {
          smooth_params_ = lerp_motion(smooth_params_, raw_params, smooth_alpha_);
        }
        correction_raw_params = raw_params;
        correction_smooth_params = smooth_params_;
      }
      corr_params.tx = correction_smooth_params.tx - correction_raw_params.tx;
      corr_params.ty = correction_smooth_params.ty - correction_raw_params.ty;
      corr_params.angle_deg = correction_smooth_params.angle_deg - correction_raw_params.angle_deg;
      const double base_scale = std::max(correction_raw_params.scale, 1e-6);
      corr_params.scale = correction_smooth_params.scale / base_scale;

      const cv::Mat correction_affine = correction_affine_2x3(
          correction_raw_params, correction_smooth_params, static_cast<int>(frame_width), static_cast<int>(frame_height));
      try {
        if (motion_model_ == MotionModel::Affine) {
          cv::warpAffine(src_bgra, stabilized, correction_affine,
                         cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)), cv::INTER_LINEAR,
                         cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));
        } else {
          const cv::Mat correction_h = affine_2x3_to_homography_3x3(correction_affine);
          cv::warpPerspective(src_bgra, stabilized, correction_h,
                              cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)), cv::INTER_LINEAR,
                              cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));
        }
      } catch (const cv::Exception& ex) {
        ++monitor_fail_frames_;
        ++consecutive_failures_;
        publish_error_if_changed(std::string("opencv warp failed: ") + ex.what(), "runtime",
                                 json::object());
        stabilized = src_bgra.clone();
        motion_valid = false;
      }
    }

    if (!motion_valid) {
      if (!scene_changed) {
        ++monitor_fail_frames_;
        ++consecutive_failures_;
        if (consecutive_failures_ >= reset_on_failure_frames_) {
          reset_stabilizer_internal(json::object(), "consecutive_failures");
        }
      }
    } else {
      consecutive_failures_ = 0;
      publish_error_if_changed("", "runtime", json::object());
    }

    prev_gray_ = gray;
    has_prev_gray_ = true;
    prev_frame_width_ = static_cast<int>(frame_width);
    prev_frame_height_ = static_cast<int>(frame_height);
  }

  if (!output_zenoh_video_ || !output_zenoh_video_->valid()) {
    ++monitor_fail_frames_;
    publish_error_if_changed("output zenoh publisher unavailable: " + output_stream_key_, "runtime", json::object());
    return;
  }
  f8::cppsdk::VideoFrameView frame;
  frame.width = frame_width;
  frame.height = frame_height;
  frame.pitch = static_cast<unsigned>(stabilized.step[0]);
  frame.format = f8::cppsdk::kVideoFormatBgra32;
  frame.frame_id = ++output_frame_id_;
  frame.ts_ms = f8::cppsdk::now_ms();
  frame.payload = reinterpret_cast<const std::byte*>(stabilized.data);
  frame.payload_bytes = stabilized.step[0] * static_cast<std::size_t>(stabilized.rows);
  if (!output_zenoh_video_->publish_frame(frame)) {
    ++monitor_fail_frames_;
    publish_error_if_changed("output zenoh publish failed: " + output_stream_key_, "runtime", json::object());
    return;
  }
  publish_state_if_changed("videoFormat", "bgra32", "runtime", json::object());
  publish_state_if_changed("videoFrameSchemaVersion", 1, "runtime", json::object());

  json motion = json::object();
  motion["frameId"] = source_frame_id;
  motion["tsMs"] = source_ts_ms;
  motion["width"] = frame_width;
  motion["height"] = frame_height;
  motion["model"] = motion_model_state_;
  motion["stabilizationMode"] = stabilization_mode_state_;
  motion["valid"] = motion_valid;
  motion["sceneChanged"] = scene_changed;
  motion["sceneChangeCount"] = scene_change_count_;
  motion["sceneCutFrameDiff"] = scene_cut_frame_diff;
  motion["sceneCutTrackRatio"] = scene_cut_track_ratio;
  motion["inlierCount"] = inlier_count;
  motion["trackedPoints"] = tracked_points;
  motion["rawTx"] = raw_params.tx;
  motion["rawTy"] = raw_params.ty;
  motion["rawAngleDeg"] = raw_params.angle_deg;
  motion["rawScale"] = raw_params.scale;
  motion["smoothTx"] = correction_smooth_params.tx;
  motion["smoothTy"] = correction_smooth_params.ty;
  motion["smoothAngleDeg"] = correction_smooth_params.angle_deg;
  motion["smoothScale"] = correction_smooth_params.scale;
  motion["corrTx"] = corr_params.tx;
  motion["corrTy"] = corr_params.ty;
  motion["corrAngleDeg"] = corr_params.angle_deg;
  motion["corrScale"] = corr_params.scale;
  motion["trajRawTx"] = trajectory_raw_params_.tx;
  motion["trajRawTy"] = trajectory_raw_params_.ty;
  motion["trajRawAngleDeg"] = trajectory_raw_params_.angle_deg;
  motion["trajRawScale"] = trajectory_raw_params_.scale;
  motion["trajSmoothTx"] = trajectory_smooth_params_.tx;
  motion["trajSmoothTy"] = trajectory_smooth_params_.ty;
  motion["trajSmoothAngleDeg"] = trajectory_smooth_params_.angle_deg;
  motion["trajSmoothScale"] = trajectory_smooth_params_.scale;
  (void)bus_->emit_data(cfg_.service_id, "motion", motion);

  const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
  emit_monitor_snapshot(end_ts_ms, source_frame_id, static_cast<double>(end_ts_ms - process_start_ms),
                        service_runtime::latency_ms_from_timestamps(end_ts_ms, source_ts_ms));
}

json VideoStabService::describe() {
  const json motion_schema = schema_object(json{{"frameId", schema_integer()},
                                                {"tsMs", schema_integer()},
                                                {"width", schema_integer()},
                                                {"height", schema_integer()},
                                                {"model", schema_string()},
                                                {"stabilizationMode", schema_string()},
                                                {"valid", schema_boolean()},
                                                {"sceneChanged", schema_boolean()},
                                                {"sceneChangeCount", schema_integer()},
                                                {"sceneCutFrameDiff", schema_number()},
                                                {"sceneCutTrackRatio", schema_number()},
                                                {"inlierCount", schema_integer()},
                                                {"trackedPoints", schema_integer()},
                                                {"rawTx", schema_number()},
                                                {"rawTy", schema_number()},
                                                {"rawAngleDeg", schema_number()},
                                                {"rawScale", schema_number()},
                                                {"smoothTx", schema_number()},
                                                {"smoothTy", schema_number()},
                                                {"smoothAngleDeg", schema_number()},
                                                {"smoothScale", schema_number()},
                                                {"corrTx", schema_number()},
                                                {"corrTy", schema_number()},
                                                {"corrAngleDeg", schema_number()},
                                                {"corrScale", schema_number()},
                                                {"trajRawTx", schema_number()},
                                                {"trajRawTy", schema_number()},
                                                {"trajRawAngleDeg", schema_number()},
                                                {"trajRawScale", schema_number()},
                                                {"trajSmoothTx", schema_number()},
                                                {"trajSmoothTy", schema_number()},
                                                {"trajSmoothAngleDeg", schema_number()},
                                                {"trajSmoothScale", schema_number()}});

  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.videostab";
  service["label"] = "CVKit Video Stabilizer";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"cv", "stabilization", "video"});

  service["stateFields"] = json::array({
      state_field("videoFormat", schema_string_enum({"bgra32"}), "ro", "Video Format", "Output video payload format.",
                  false),
      state_field("videoFrameSchemaVersion", schema_integer(1, 1, 1), "ro", "Video Frame Schema",
                  "Output video frame schema version.", false),
      state_field("motionModel", schema_string_enum({"affine", "homography"}, "affine"), "rw", "Motion Model",
                  "Global motion model used by stabilizer.", false),
      state_field("stabilizationMode", schema_string_enum({"trajectory", "instant"}, "trajectory"), "rw",
                  "Stabilization Mode", "trajectory=smooth accumulated path; instant=smooth per-frame motion.", false),
      state_field("smoothAlpha", schema_number(0.15, 0.01, 0.5), "rw", "Smooth Alpha",
                  "EMA alpha used for motion smoothing.", false, json{{"kind", "slider"}}),
      state_field("maxCornerCount", schema_integer(300, 20, 2000), "rw", "Max Corner Count", "LK feature count."),
      state_field("qualityLevel", schema_number(0.01, 0.0001, 0.3), "rw", "Quality Level",
                  "goodFeaturesToTrack quality level."),
      state_field("minDistance", schema_number(8.0, 1.0, 100.0), "rw", "Min Distance", "Minimum corner distance."),
      state_field("ransacReprojThreshold", schema_number(3.0, 0.1, 20.0), "rw", "RANSAC Threshold",
                  "RANSAC reprojection threshold."),
      state_field("resetOnFailureFrames", schema_integer(5, 1, 120), "rw", "Reset On Failure Frames",
                  "Reset internal stabilizer state after N consecutive failures.", false),
      state_field("sceneCutEnabled", schema_boolean(), "rw", "Scene Cut Enabled",
                  "Enable scene cut detection and reset-on-cut behavior.", false),
      state_field("sceneCutFrameDiffThreshold", schema_number(18.0, 1.0, 80.0), "rw", "Cut Frame Diff Threshold",
                  "Scene cut threshold for mean(abs(gray-prevGray)).", false),
      state_field("sceneCutTrackRatioThreshold", schema_number(0.25, 0.01, 0.95), "rw", "Cut Track Ratio Threshold",
                  "Scene cut threshold for trackedPoints/max(prevPoints,1).", false),
      state_field("sceneCutCooldownFrames", schema_integer(5, 0, 120), "rw", "Cut Cooldown Frames",
                  "Suppress repeated scene cut triggers for N frames after a cut.", false),
  });

  service["commands"] = json::array({
      json{{"name", "resetStabilizer"},
           {"description", "Reset internal trajectory/smoothing state."},
           {"required", true},
           {"showOnNode", true}},
  });

  service["dataInPorts"] = json::array({
      video_frame_port("video", "Input video frame stream."),
  });
  service["dataOutPorts"] = json::array({
      video_frame_port("video", "Stabilized video frame stream."),
      json{{"name", "motion"},
           {"valueSchema", motion_schema},
           {"description", "Per-frame estimated and smoothed motion parameters."},
           {"required", true},
           {"showOnNode", true}},
  });

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::video_stab
