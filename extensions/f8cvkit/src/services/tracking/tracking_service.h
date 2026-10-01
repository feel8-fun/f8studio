#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/service_bus.h"

namespace f8::cvkit::tracking {

struct TrackingInitCandidate {
  cv::Rect bbox;
  std::optional<double> score;
};

enum class TrackingInitSelectMode {
  FirstBox,
  ClosestCenter,
  LargestArea,
  HighestScore,
};

enum class TrackerKind {
  Csrt,
  Kcf,
  Mil,
  Nano,
  Vit,
};

class TrackingService final : public f8::cppsdk::LifecycleNode,
                              public f8::cppsdk::StatefulNode,
                              public f8::cppsdk::DataReceivableNode,
                              public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.tracking";
    f8::cppsdk::RuntimeBackendConfig runtime_backend;
    std::string tracker_kind = "csrt";
    std::string model_dir;
    bool auto_download_models = true;
    double max_tracking_fps = 30.0;
    int stop_tracking_cooldown_ms = 1000;
  };

  explicit TrackingService(Config cfg);
  ~TrackingService();

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
  void publish_error_if_changed(const json& value, const std::string& source, const json& meta);
  void emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms, double latency_ms);
  void set_init_select(const std::string& mode, const json& meta);
  void set_tracker_kind(const std::string& kind, const json& meta);
  void set_model_dir(const std::string& model_dir, const json& meta);
  void set_max_tracking_fps(double fps, const json& meta);
  bool ensure_zenoh_video_open();
  bool copy_latest_video_frame(std::vector<std::byte>& out_payload, f8::cppsdk::LatestVideoFrame& out_frame,
                               bool changed_only, std::uint64_t last_frame_id,
                               std::chrono::milliseconds timeout);
  void apply_init_box_if_any();
  void process_frame_once();
  void set_tracking(bool tracking, const json& meta);
  void stop_tracking_internal(const json& meta);

  Config cfg_;
  std::atomic<int> stop_tracking_cooldown_ms_{1000};
  std::atomic<std::int64_t> stop_tracking_cooldown_until_ms_{0};
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};
  std::atomic<double> max_tracking_fps_{30.0};
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  // Video input.
  f8::cppsdk::ZenohLatestVideoFrameSubscriber zenoh_video_;
  std::string zenoh_video_open_key_;
  std::vector<std::byte> frame_bgra_;
  cv::Mat frame_bgr_;
  std::uint64_t last_frame_id_ = 0;
  int tracking_frame_width_ = 0;
  int tracking_frame_height_ = 0;
  std::int64_t last_processed_frame_ts_ms_ = 0;
  double next_tracking_due_ts_ms_ = 0.0;
  std::int64_t last_video_open_attempt_ms_ = 0;
  std::int64_t init_video_wait_started_ms_ = 0;
  std::int64_t init_video_wait_last_log_ms_ = 0;
  std::uint32_t init_video_wait_misses_ = 0;
  std::uint64_t init_box_messages_seen_ = 0;
  std::uint64_t init_box_no_candidate_messages_ = 0;
  std::uint64_t init_video_wait_error_generation_ = 0;
  bool init_video_wait_error_active_ = false;
  TrackingInitSelectMode init_select_mode_ = TrackingInitSelectMode::ClosestCenter;
  std::string init_select_state_ = "closest_center";
  TrackerKind tracker_kind_ = TrackerKind::Csrt;
  std::string tracker_kind_state_ = "csrt";
  std::string active_tracker_kind_state_;
  std::string model_dir_state_ = "models";
  std::filesystem::path model_dir_path_;
  bool auto_download_models_ = true;
  std::int64_t model_download_retry_after_ms_ = 0;

  // Tracking state.
  std::mutex tracking_mu_;
  cv::Ptr<cv::Tracker> tracker_;
  cv::Rect bbox_;
  bool is_tracking_ = false;

  // Pending init candidates extracted from upstream payloads.
  std::vector<TrackingInitCandidate> pending_init_boxes_;
  std::uint64_t pending_init_box_generation_ = 0;

  std::uint64_t monitor_observed_frames_ = 0;
  std::uint64_t monitor_processed_frames_ = 0;
  std::uint64_t monitor_window_processed_frames_ = 0;
  std::int64_t monitor_window_start_ms_ = 0;
  double monitor_last_process_ms_ = 0.0;
  double monitor_total_process_ms_ = 0.0;
  double monitor_last_latency_ms_ = 0.0;
  double monitor_total_latency_ms_ = 0.0;
  double monitor_fps_ = 0.0;
};

}  // namespace f8::cvkit::tracking
