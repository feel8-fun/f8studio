#include "video_stab_service.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <exception>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/shm/naming.h"
#include "f8cppsdk/shm/sizing.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cvkit::video_stab {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_boolean() { return json{{"type", "boolean"}}; }
json schema_integer() { return json{{"type", "integer"}}; }
json schema_number() { return json{{"type", "number"}}; }
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
json schema_string_enum(const std::vector<std::string>& values, const std::string& default_value) {
  json s{{"type", "string"}};
  s["enum"] = json::array();
  for (const std::string& v : values) {
    s["enum"].push_back(v);
  }
  s["default"] = default_value;
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

std::string trim_copy(const std::string& s) {
  std::string out = s;
  while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
  while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
  return out;
}

std::string to_lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

int count_inliers(const cv::Mat& inlier_mask) {
  if (inlier_mask.empty()) return 0;
  if (inlier_mask.type() != CV_8UC1) return 0;
  return cv::countNonZero(inlier_mask);
}

VideoStabService::MotionParams motion_from_affine(const cv::Mat& affine_2x3) {
  VideoStabService::MotionParams out;
  if (affine_2x3.empty() || affine_2x3.rows != 2 || affine_2x3.cols != 3) return out;
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
  if (homography_3x3.empty() || homography_3x3.rows != 3 || homography_3x3.cols != 3) return out;
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
  if (out.scale <= 1e-6) out.scale = 1.0;
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

VideoStabService::~VideoStabService() { stop(); }

bool VideoStabService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Video Stabilizer";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);
  bus_->add_command_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

      output_shm_name_ = f8::cppsdk::shm::video_shm_name(cfg_.service_id);
  input_shm_name_.clear();
  input_video_.close();
  output_initialized_ = false;
  has_prev_gray_ = false;
  prev_gray_.release();
  smooth_initialized_ = false;
  smooth_params_ = MotionParams{};
  trajectory_initialized_ = false;
  trajectory_raw_params_ = MotionParams{};
  trajectory_smooth_params_ = MotionParams{};
  consecutive_failures_ = 0;
  scene_change_count_ = 0;
  scene_cut_cooldown_remaining_ = 0;

  input_last_notify_seq_ = 0;
  input_last_frame_id_ = 0;
  input_last_open_attempt_ms_ = 0;
  output_last_open_attempt_ms_ = 0;

  telemetry_observed_frames_ = 0;
  telemetry_processed_frames_ = 0;
  telemetry_window_processed_frames_ = 0;
  telemetry_fail_frames_ = 0;
  telemetry_window_start_ms_ = 0;
  telemetry_last_process_ms_ = 0.0;
  telemetry_total_process_ms_ = 0.0;
  telemetry_fps_ = 0.0;

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("inputShmName", input_shm_name_, "init", json::object());
  publish_state_if_changed("outputShmName", output_shm_name_, "init", json::object());
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
  publish_state_if_changed("sceneChangeCount", scene_change_count_, "init", json::object());
  publish_state_if_changed("lastError", "", "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_video_stab started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void VideoStabService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();

  std::lock_guard<std::mutex> lock(io_mu_);
  input_video_.close();
  output_video_.reset();
  output_initialized_ = false;
}

void VideoStabService::tick() {
  if (!running()) return;
  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
  }
  if (!active_.load(std::memory_order_acquire)) return;
  process_frame_once();
}

void VideoStabService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                                const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) {
    (void)f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, value, source, meta);
  }
}

void VideoStabService::emit_telemetry(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms) {
  if (!bus_) return;
  if (telemetry_window_start_ms_ <= 0) {
    telemetry_window_start_ms_ = ts_ms;
  }
  ++telemetry_processed_frames_;
  ++telemetry_window_processed_frames_;
  telemetry_last_process_ms_ = process_ms;
  telemetry_total_process_ms_ += process_ms;

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
  telemetry["failFrames"] = telemetry_fail_frames_;
  (void)bus_->emit_data(cfg_.service_id, "telemetry", telemetry);
}

void VideoStabService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

bool VideoStabService::parse_double_field(const json& value, double& out) const {
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

bool VideoStabService::parse_int_field(const json& value, int& out) const {
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

void VideoStabService::set_input_shm_name(const std::string& shm_name, const json& meta) {
  const std::string trimmed = trim_copy(shm_name);
  if (trimmed == input_shm_name_) {
    publish_state_if_changed("inputShmName", input_shm_name_, "state", meta);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(io_mu_);
    input_shm_name_ = trimmed;
    input_video_.close();
    input_last_open_attempt_ms_ = 0;
    input_last_notify_seq_ = 0;
    input_last_frame_id_ = 0;
  }

  reset_stabilizer_internal(meta, "input_shm_changed");
  publish_state_if_changed("inputShmName", input_shm_name_, "state", meta);
}

void VideoStabService::set_motion_model(const std::string& model, const json& meta) {
  const std::string normalized = to_lower_copy(trim_copy(model));
  if (normalized == "affine") {
    motion_model_ = MotionModel::Affine;
    motion_model_state_ = "affine";
  } else if (normalized == "homography") {
    motion_model_ = MotionModel::Homography;
    motion_model_state_ = "homography";
  } else {
    publish_state_if_changed("lastError", "invalid motionModel: " + model, "state", meta);
    return;
  }
  reset_stabilizer_internal(meta, "motion_model_changed");
  publish_state_if_changed("motionModel", motion_model_state_, "state", meta);
  publish_state_if_changed("lastError", "", "state", meta);
}

void VideoStabService::set_stabilization_mode(const std::string& mode, const json& meta) {
  const std::string normalized = to_lower_copy(trim_copy(mode));
  if (normalized == "trajectory") {
    stabilization_mode_ = StabilizationMode::Trajectory;
    stabilization_mode_state_ = "trajectory";
  } else if (normalized == "instant") {
    stabilization_mode_ = StabilizationMode::Instant;
    stabilization_mode_state_ = "instant";
  } else {
    publish_state_if_changed("lastError", "invalid stabilizationMode: " + mode, "state", meta);
    return;
  }
  reset_stabilizer_internal(meta, "stabilization_mode_changed");
  publish_state_if_changed("stabilizationMode", stabilization_mode_state_, "state", meta);
  publish_state_if_changed("lastError", "", "state", meta);
}

void VideoStabService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;

  if (field == "inputShmName" && value.is_string()) {
    set_input_shm_name(value.get<std::string>(), meta);
    return;
  }
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
      publish_state_if_changed("lastError", "invalid smoothAlpha", "state", meta);
      return;
    }
    smooth_alpha_ = std::max(0.01, std::min(0.5, v));
    publish_state_if_changed("smoothAlpha", smooth_alpha_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "maxCornerCount") {
    int v = 0;
    if (!parse_int_field(value, v)) {
      publish_state_if_changed("lastError", "invalid maxCornerCount", "state", meta);
      return;
    }
    max_corner_count_ = std::max(20, std::min(2000, v));
    publish_state_if_changed("maxCornerCount", max_corner_count_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "qualityLevel") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_state_if_changed("lastError", "invalid qualityLevel", "state", meta);
      return;
    }
    quality_level_ = std::max(0.0001, std::min(0.3, v));
    publish_state_if_changed("qualityLevel", quality_level_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "minDistance") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_state_if_changed("lastError", "invalid minDistance", "state", meta);
      return;
    }
    min_distance_ = std::max(1.0, std::min(100.0, v));
    publish_state_if_changed("minDistance", min_distance_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "ransacReprojThreshold") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_state_if_changed("lastError", "invalid ransacReprojThreshold", "state", meta);
      return;
    }
    ransac_reproj_threshold_ = std::max(0.1, std::min(20.0, v));
    publish_state_if_changed("ransacReprojThreshold", ransac_reproj_threshold_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "resetOnFailureFrames") {
    int v = 0;
    if (!parse_int_field(value, v)) {
      publish_state_if_changed("lastError", "invalid resetOnFailureFrames", "state", meta);
      return;
    }
    reset_on_failure_frames_ = std::max(1, std::min(120, v));
    publish_state_if_changed("resetOnFailureFrames", reset_on_failure_frames_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "sceneCutEnabled") {
    if (!value.is_boolean()) {
      publish_state_if_changed("lastError", "invalid sceneCutEnabled", "state", meta);
      return;
    }
    scene_cut_enabled_ = value.get<bool>();
    publish_state_if_changed("sceneCutEnabled", scene_cut_enabled_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "sceneCutFrameDiffThreshold") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_state_if_changed("lastError", "invalid sceneCutFrameDiffThreshold", "state", meta);
      return;
    }
    scene_cut_frame_diff_threshold_ = std::max(1.0, std::min(80.0, v));
    publish_state_if_changed("sceneCutFrameDiffThreshold", scene_cut_frame_diff_threshold_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "sceneCutTrackRatioThreshold") {
    double v = 0.0;
    if (!parse_double_field(value, v)) {
      publish_state_if_changed("lastError", "invalid sceneCutTrackRatioThreshold", "state", meta);
      return;
    }
    scene_cut_track_ratio_threshold_ = std::max(0.01, std::min(0.95, v));
    publish_state_if_changed("sceneCutTrackRatioThreshold", scene_cut_track_ratio_threshold_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
    return;
  }

  if (field == "sceneCutCooldownFrames") {
    int v = 0;
    if (!parse_int_field(value, v)) {
      publish_state_if_changed("lastError", "invalid sceneCutCooldownFrames", "state", meta);
      return;
    }
    scene_cut_cooldown_frames_ = std::max(0, std::min(120, v));
    publish_state_if_changed("sceneCutCooldownFrames", scene_cut_cooldown_frames_, "state", meta);
    publish_state_if_changed("lastError", "", "state", meta);
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
  // Pull-only service, no dataIn.
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
  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (input_video_.readHeader(hdr)) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (input_last_open_attempt_ms_ > 0 && (now - input_last_open_attempt_ms_) < 1000) {
    return false;
  }
  input_last_open_attempt_ms_ = now;

  if (input_shm_name_.empty()) {
    publish_state_if_changed("lastError", "missing inputShmName", "runtime", json::object());
    return false;
  }

  if (!input_video_.open(input_shm_name_, f8::cppsdk::shm::kDefaultVideoShmBytes)) {
    publish_state_if_changed("lastError", "video shm open failed: " + input_shm_name_, "runtime", json::object());
    return false;
  }

  input_last_notify_seq_ = 0;
  publish_state_if_changed("lastError", "", "runtime", json::object());
  return true;
}

bool VideoStabService::ensure_output_open() {
  if (output_initialized_) return true;

  const std::int64_t now = f8::cppsdk::now_ms();
  if (output_last_open_attempt_ms_ > 0 && (now - output_last_open_attempt_ms_) < 1000) {
    return false;
  }
  output_last_open_attempt_ms_ = now;

  output_video_ = std::make_unique<f8::cppsdk::VideoSharedMemorySink>();
  if (!output_video_->initialize(output_shm_name_, f8::cppsdk::shm::kDefaultVideoShmBytes,
                                 f8::cppsdk::shm::kDefaultVideoShmSlots)) {
    output_video_.reset();
    publish_state_if_changed("lastError", "output shm init failed: " + output_shm_name_, "runtime", json::object());
    return false;
  }

  output_initialized_ = true;
  publish_state_if_changed("lastError", "", "runtime", json::object());
  return true;
}

void VideoStabService::process_frame_once() {
  if (!bus_) return;

  std::lock_guard<std::mutex> lock(io_mu_);

  if (!ensure_input_open()) {
    return;
  }
  if (!ensure_output_open()) {
    return;
  }

  std::uint32_t observed_notify_seq = input_last_notify_seq_;
  if (!input_video_.waitNewFrame(input_last_notify_seq_, 20, &observed_notify_seq)) {
    return;
  }
  input_last_notify_seq_ = observed_notify_seq;

  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (!input_video_.copyLatestFrame(input_frame_bgra_, hdr)) {
    return;
  }
  if (hdr.frame_id == 0 || hdr.frame_id == input_last_frame_id_) {
    return;
  }

  ++telemetry_observed_frames_;
  input_last_frame_id_ = hdr.frame_id;
  const std::int64_t process_start_ms = f8::cppsdk::now_ms();

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
  if (input_frame_bgra_.size() < row_bytes * static_cast<std::size_t>(hdr.height)) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "video shm frame too small", "runtime", json::object());
    return;
  }

  cv::Mat src_bgra(static_cast<int>(hdr.height), static_cast<int>(hdr.width), CV_8UC4,
                   const_cast<std::byte*>(input_frame_bgra_.data()), row_bytes);

  cv::Mat gray;
  try {
    cv::cvtColor(src_bgra, gray, cv::COLOR_BGRA2GRAY);
  } catch (const cv::Exception& ex) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", std::string("opencv cvtColor failed: ") + ex.what(), "runtime", json::object());
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
      ++telemetry_fail_frames_;
      ++consecutive_failures_;
      publish_state_if_changed("lastError", std::string("opencv optical flow failed: ") + ex.what(), "runtime",
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
    const int w = static_cast<int>(hdr.width);
    const int h = static_cast<int>(hdr.height);

    for (std::size_t i = 0; i < prev_pts.size() && i < curr_pts.size() && i < status.size(); ++i) {
      if (status[i] == 0) continue;
      const cv::Point2f p0 = prev_pts[i];
      const cv::Point2f p1 = curr_pts[i];
      if (p0.x < 0.0f || p0.y < 0.0f || p0.x >= static_cast<float>(w) || p0.y >= static_cast<float>(h)) continue;
      if (p1.x < 0.0f || p1.y < 0.0f || p1.x >= static_cast<float>(w) || p1.y >= static_cast<float>(h)) continue;
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
      publish_state_if_changed("sceneChangeCount", scene_change_count_, "runtime", json::object());
      reset_stabilizer_internal(json::object(), "scene_cut");
      scene_cut_cooldown_remaining_ = scene_cut_cooldown_frames_;
      prev_gray_ = gray;
      has_prev_gray_ = true;
      consecutive_failures_ = 0;
      publish_state_if_changed("lastError", "", "runtime", json::object());
    } else if (tracked_points >= 8) {
      cv::Mat inliers;
      try {
        if (motion_model_ == MotionModel::Affine) {
          cv::Mat affine = cv::estimateAffinePartial2D(prev_valid, curr_valid, inliers, cv::RANSAC,
                                                       ransac_reproj_threshold_);
          if (!affine.empty() && affine.rows == 2 && affine.cols == 3) {
            raw_params = motion_from_affine(affine);
            inlier_count = count_inliers(inliers);
            motion_valid = true;
          }
        } else {
          cv::Mat homography = cv::findHomography(prev_valid, curr_valid, cv::RANSAC, ransac_reproj_threshold_, inliers);
          if (!homography.empty() && homography.rows == 3 && homography.cols == 3) {
            raw_params = motion_from_homography(homography);
            inlier_count = count_inliers(inliers);
            motion_valid = true;
          }
        }
      } catch (const cv::Exception& ex) {
        ++telemetry_fail_frames_;
        ++consecutive_failures_;
        publish_state_if_changed("lastError", std::string("opencv transform estimate failed: ") + ex.what(), "runtime",
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
          trajectory_smooth_params_ =
              lerp_motion(trajectory_smooth_params_, trajectory_raw_params_, smooth_alpha_);
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

      const cv::Mat correction_affine =
          correction_affine_2x3(correction_raw_params, correction_smooth_params, static_cast<int>(hdr.width),
                                static_cast<int>(hdr.height));
      try {
        if (motion_model_ == MotionModel::Affine) {
          cv::warpAffine(src_bgra, stabilized, correction_affine,
                         cv::Size(static_cast<int>(hdr.width), static_cast<int>(hdr.height)), cv::INTER_LINEAR,
                         cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));
        } else {
          const cv::Mat correction_h = affine_2x3_to_homography_3x3(correction_affine);
          cv::warpPerspective(src_bgra, stabilized, correction_h,
                              cv::Size(static_cast<int>(hdr.width), static_cast<int>(hdr.height)), cv::INTER_LINEAR,
                              cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));
        }
      } catch (const cv::Exception& ex) {
        ++telemetry_fail_frames_;
        ++consecutive_failures_;
        publish_state_if_changed("lastError", std::string("opencv warp failed: ") + ex.what(), "runtime", json::object());
        stabilized = src_bgra.clone();
        motion_valid = false;
      }
    }

    if (!motion_valid) {
      if (!scene_changed) {
        ++telemetry_fail_frames_;
        ++consecutive_failures_;
        if (consecutive_failures_ >= reset_on_failure_frames_) {
          reset_stabilizer_internal(json::object(), "consecutive_failures");
        }
      }
    } else {
      consecutive_failures_ = 0;
      publish_state_if_changed("lastError", "", "runtime", json::object());
    }

    prev_gray_ = gray;
    has_prev_gray_ = true;
  }

  if (!output_video_ || !output_video_->ensureConfiguration(hdr.width, hdr.height)) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "output shm ensureConfiguration failed", "runtime", json::object());
    return;
  }

  if (!output_video_->writeFrame(stabilized.data, static_cast<unsigned>(stabilized.step[0]))) {
    ++telemetry_fail_frames_;
    publish_state_if_changed("lastError", "output shm writeFrame failed", "runtime", json::object());
    return;
  }

  json motion = json::object();
  motion["frameId"] = hdr.frame_id;
  motion["tsMs"] = hdr.ts_ms;
  motion["width"] = hdr.width;
  motion["height"] = hdr.height;
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
  emit_telemetry(end_ts_ms, hdr.frame_id, static_cast<double>(end_ts_ms - process_start_ms));
}

json VideoStabService::describe() {
  const json motion_schema = schema_object(
      json{{"frameId", schema_integer()},
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

  const json telemetry_schema = schema_object(
      json{{"tsMs", schema_integer()},
           {"frameId", schema_integer()},
           {"fps", schema_number()},
           {"processMs", schema_number()},
           {"avgProcessMs", schema_number()},
           {"observedFrames", schema_integer()},
           {"processedFrames", schema_integer()},
           {"droppedFrames", schema_integer()},
           {"failFrames", schema_integer()}});

  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.videostab";
  service["label"] = "CVKit Video Stabilizer";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"cv", "stabilization", "video"});

  service["stateFields"] = json::array({
      state_field("inputShmName", schema_string(), "rw", "Input Video SHM", "Input SHM name (e.g. shm.xxx.video).", true),
      state_field("outputShmName", schema_string(), "ro", "Output Video SHM", "Output SHM name generated from serviceId.",
                  true),
      state_field("motionModel", schema_string_enum({"affine", "homography"}, "affine"), "rw", "Motion Model",
                  "Global motion model used by stabilizer.", false),
      state_field("stabilizationMode", schema_string_enum({"trajectory", "instant"}, "trajectory"), "rw",
                  "Stabilization Mode", "trajectory=smooth accumulated path; instant=smooth per-frame motion.", false),
      state_field("smoothAlpha", schema_number(0.15, 0.01, 0.5), "rw", "Smooth Alpha",
                  "EMA alpha used for motion smoothing.", false, "slider"),
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
      state_field("sceneChangeCount", schema_integer(), "ro", "Scene Change Count",
                  "Monotonic counter incremented when a scene cut is detected.", false),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message.", false),
  });
  service["editableStateFields"] = false;

  service["commands"] = json::array({
      json{{"name", "resetStabilizer"},
           {"description", "Reset internal trajectory/smoothing state."},
           {"showOnNode", true}},
  });
  service["editableCommands"] = false;

  service["dataInPorts"] = json::array();
  service["dataOutPorts"] = json::array({
      json{{"name", "motion"},
           {"valueSchema", motion_schema},
           {"description", "Per-frame estimated and smoothed motion parameters."},
           {"required", false}},
      json{{"name", "telemetry"},
           {"valueSchema", telemetry_schema},
           {"description", "Runtime telemetry: fps/process time/dropped/fail frames."},
           {"required", false}},
  });
  service["editableDataInPorts"] = false;
  service["editableDataOutPorts"] = false;

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::video_stab
