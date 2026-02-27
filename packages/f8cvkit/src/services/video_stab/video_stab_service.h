#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include <opencv2/core.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/video_shared_memory_sink.h"

namespace f8::cvkit::video_stab {

enum class MotionModel {
  Affine,
  Homography,
};

enum class StabilizationMode {
  Trajectory,
  Instant,
};

class VideoStabService final : public f8::cppsdk::LifecycleNode,
                               public f8::cppsdk::StatefulNode,
                               public f8::cppsdk::DataReceivableNode,
                               public f8::cppsdk::CommandableNode {
 public:
  struct MotionParams {
    double tx = 0.0;
    double ty = 0.0;
    double angle_deg = 0.0;
    double scale = 1.0;
  };

  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.videostab";
    std::string nats_url = "nats://127.0.0.1:4222";
  };

  explicit VideoStabService(Config cfg);
  ~VideoStabService();

  bool start();
  void stop();
  bool running() const {
    return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
  }

  void tick();

  void on_lifecycle(bool active, const nlohmann::json& meta) override;
  void on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value, std::int64_t ts_ms,
                const nlohmann::json& meta) override;
  void on_data(const std::string& node_id, const std::string& port, const nlohmann::json& value, std::int64_t ts_ms,
               const nlohmann::json& meta) override;
  bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                  nlohmann::json& result, std::string& error_code, std::string& error_message) override;

  static nlohmann::json describe();

 private:
  using json = nlohmann::json;

  void publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                const json& meta);
  void emit_telemetry(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms);
  bool ensure_input_open();
  bool ensure_output_open();
  void process_frame_once();
  void reset_stabilizer_internal(const json& meta, const std::string& reason);
  void set_input_shm_name(const std::string& shm_name, const json& meta);
  void set_motion_model(const std::string& model, const json& meta);
  void set_stabilization_mode(const std::string& mode, const json& meta);

  bool parse_double_field(const json& value, double& out) const;
  bool parse_int_field(const json& value, int& out) const;

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  std::mutex io_mu_;

  // Input video SHM (BGRA32).
  std::string input_shm_name_;
  f8::cppsdk::VideoSharedMemoryReader input_video_;
  std::vector<std::byte> input_frame_bgra_;
  std::uint32_t input_last_notify_seq_ = 0;
  std::uint64_t input_last_frame_id_ = 0;
  std::int64_t input_last_open_attempt_ms_ = 0;

  // Output video SHM (BGRA32).
  std::string output_shm_name_;
  std::unique_ptr<f8::cppsdk::VideoSharedMemorySink> output_video_;
  bool output_initialized_ = false;
  std::int64_t output_last_open_attempt_ms_ = 0;

  // Stabilizer tuning.
  MotionModel motion_model_ = MotionModel::Affine;
  std::string motion_model_state_ = "affine";
  StabilizationMode stabilization_mode_ = StabilizationMode::Trajectory;
  std::string stabilization_mode_state_ = "trajectory";
  double smooth_alpha_ = 0.15;
  int max_corner_count_ = 300;
  double quality_level_ = 0.01;
  double min_distance_ = 8.0;
  double ransac_reproj_threshold_ = 3.0;
  int reset_on_failure_frames_ = 5;
  bool scene_cut_enabled_ = true;
  double scene_cut_frame_diff_threshold_ = 18.0;
  double scene_cut_track_ratio_threshold_ = 0.25;
  int scene_cut_cooldown_frames_ = 5;
  std::uint64_t scene_change_count_ = 0;

  // Tracking state.
  bool has_prev_gray_ = false;
  cv::Mat prev_gray_;
  bool smooth_initialized_ = false;
  MotionParams smooth_params_{};
  bool trajectory_initialized_ = false;
  MotionParams trajectory_raw_params_{};
  MotionParams trajectory_smooth_params_{};
  int consecutive_failures_ = 0;
  int scene_cut_cooldown_remaining_ = 0;

  std::uint64_t telemetry_observed_frames_ = 0;
  std::uint64_t telemetry_processed_frames_ = 0;
  std::uint64_t telemetry_window_processed_frames_ = 0;
  std::uint64_t telemetry_fail_frames_ = 0;
  std::int64_t telemetry_window_start_ms_ = 0;
  double telemetry_last_process_ms_ = 0.0;
  double telemetry_total_process_ms_ = 0.0;
  double telemetry_fps_ = 0.0;
};

}  // namespace f8::cvkit::video_stab
