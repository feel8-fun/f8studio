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

namespace f8::cvkit::dense_optflow {

class DenseOptflowService final : public f8::cppsdk::LifecycleNode,
                                  public f8::cppsdk::StatefulNode,
                                  public f8::cppsdk::DataReceivableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.denseoptflow";
    std::string nats_url = "nats://127.0.0.1:4222";
  };

  explicit DenseOptflowService(Config cfg);
  ~DenseOptflowService();

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

  static nlohmann::json describe();

 private:
  using json = nlohmann::json;

  void publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                const json& meta);
  void emit_telemetry(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms, std::uint64_t vectors_per_frame);

  bool ensure_video_open();
  void process_frame_once();

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  std::mutex flow_mu_;

  // Input SHM settings.
  std::string input_shm_name_;
  int compute_every_n_frames_ = 2;
  int sample_step_px_ = 16;
  double min_mag_ = 0.0;

  // Video reader state.
  f8::cppsdk::VideoSharedMemoryReader video_;
  std::vector<std::byte> frame_bgra_;
  std::uint32_t last_notify_seq_ = 0;
  std::uint64_t last_frame_id_ = 0;
  std::int64_t last_video_open_attempt_ms_ = 0;
  std::uint64_t frame_counter_ = 0;

  // Previous compute frame in grayscale.
  cv::Mat prev_gray_;
  bool has_prev_gray_ = false;
  int prev_width_ = 0;
  int prev_height_ = 0;

  // Telemetry.
  std::uint64_t telemetry_observed_frames_ = 0;
  std::uint64_t telemetry_processed_frames_ = 0;
  std::uint64_t telemetry_window_processed_frames_ = 0;
  std::uint64_t telemetry_fail_frames_ = 0;
  std::uint64_t telemetry_last_vectors_per_frame_ = 0;
  std::int64_t telemetry_window_start_ms_ = 0;
  double telemetry_last_process_ms_ = 0.0;
  double telemetry_total_process_ms_ = 0.0;
  double telemetry_fps_ = 0.0;
};

}  // namespace f8::cvkit::dense_optflow
