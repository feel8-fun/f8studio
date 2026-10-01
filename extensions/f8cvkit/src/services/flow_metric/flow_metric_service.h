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
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/service_bus.h"

namespace f8::cvkit::flow_metric {

class FlowMetricService final : public f8::cppsdk::LifecycleNode,
                                public f8::cppsdk::StatefulNode,
                                public f8::cppsdk::DataReceivableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.flowmetric";
    f8::cppsdk::RuntimeBackendConfig runtime_backend;
  };

  explicit FlowMetricService(Config cfg);
  ~FlowMetricService();

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
  enum class MetricMode {
    Divergence,
    Magnitude,
    Curl,
    Strain,
  };

  void publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                const json& meta);
  void publish_error_if_changed(const json& value, const std::string& source, const json& meta);
  void emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms, double latency_ms,
                             std::uint64_t points_per_frame);
  bool ensure_flow_open();
  void process_frame_once();

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  std::mutex io_mu_;

  // Input flow settings and reader state.
  std::string input_flow_stream_key_;
  int compute_every_n_frames_ = 1;
  MetricMode metric_mode_ = MetricMode::Divergence;
  std::string metric_mode_state_ = "divergence";
  double metric_scale_ = 1.0;
  std::unique_ptr<f8::cppsdk::ZenohLatestVideoFrameSubscriber> input_zenoh_flow_;
  std::vector<std::byte> flow_payload_;
  std::uint64_t last_frame_id_ = 0;
  std::int64_t last_flow_open_attempt_ms_ = 0;
  std::uint64_t frame_counter_ = 0;

  // Output scalar settings and writer state.
  std::string scalar_stream_key_;
  std::string scalar_format_ = "scalar1_f32";
  std::shared_ptr<f8::cppsdk::ZenohLatestVideoFramePublisher> scalar_zenoh_publisher_;
  std::vector<std::byte> scalar_payload_;
  std::uint64_t scalar_output_frame_id_ = 0;

  // Temporary compute buffers.
  cv::Mat flow_u_;
  cv::Mat flow_v_;
  cv::Mat du_dx_;
  cv::Mat du_dy_;
  cv::Mat dv_dx_;
  cv::Mat dv_dy_;
  cv::Mat metric_output_;

  // Monitor stats.
  std::uint64_t monitor_observed_frames_ = 0;
  std::uint64_t monitor_processed_frames_ = 0;
  std::uint64_t monitor_window_processed_frames_ = 0;
  std::uint64_t monitor_fail_frames_ = 0;
  std::uint64_t monitor_last_points_per_frame_ = 0;
  std::int64_t monitor_window_start_ms_ = 0;
  double monitor_last_process_ms_ = 0.0;
  double monitor_total_process_ms_ = 0.0;
  double monitor_last_latency_ms_ = 0.0;
  double monitor_total_latency_ms_ = 0.0;
  double monitor_fps_ = 0.0;
};

}  // namespace f8::cvkit::flow_metric
