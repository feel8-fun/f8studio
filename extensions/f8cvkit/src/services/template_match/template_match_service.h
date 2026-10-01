#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

#include <opencv2/core.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/latest_video_frame_transport.h"
#include "f8cppsdk/service_bus.h"

namespace f8::cvkit::template_match {

class TemplateMatchService final : public f8::cppsdk::LifecycleNode,
                                   public f8::cppsdk::StatefulNode,
                                   public f8::cppsdk::DataReceivableNode,
                                   public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.templatematch";
    f8::cppsdk::RuntimeBackendConfig runtime_backend;
  };

  explicit TemplateMatchService(Config cfg);
  ~TemplateMatchService();

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
  void set_template_png_b64(const std::string& b64, const json& meta);
  bool ensure_zenoh_video_open();
  bool copy_latest_video_frame(std::vector<std::byte>& out_payload, f8::cppsdk::LatestVideoFrame& out_frame,
                               bool changed_only, std::uint64_t last_frame_id,
                               std::chrono::milliseconds timeout);
  void detect_once();

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;
  std::mutex video_mu_;

  // Template (BGR/gray).
  bool template_loaded_ = false;
  std::string template_error_;
  cv::Mat template_bgr_;
  cv::Mat template_gray_;
  std::string template_png_b64_;
  double match_threshold_ = 0.50;
  std::int64_t matching_interval_ms_ = 200;
  std::int64_t last_match_ts_ms_ = 0;
  std::string match_color_mode_ = "gray";
  int search_roi_padding_px_ = 0;
  double pyramid_scale_ = 1.0;
  bool has_last_detection_ = false;
  cv::Rect last_detection_bbox_;

  // Video input.
  f8::cppsdk::ZenohLatestVideoFrameSubscriber zenoh_video_;
  std::string zenoh_video_open_key_;
  std::vector<std::byte> frame_bgra_;
  cv::Mat frame_gray_;
  cv::Mat roi_gray_;
  cv::Mat roi_small_;
  cv::Mat templ_small_;
  cv::Mat match_result_;
  std::uint64_t last_frame_id_ = 0;
  std::int64_t last_video_open_attempt_ms_ = 0;

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

}  // namespace f8::cvkit::template_match
