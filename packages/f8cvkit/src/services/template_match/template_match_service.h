#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

#include <opencv2/core.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/video_shared_memory_sink.h"

namespace f8::cvkit::template_match {

class TemplateMatchService final : public f8::cppsdk::LifecycleNode,
                                   public f8::cppsdk::StatefulNode,
                                   public f8::cppsdk::DataReceivableNode,
                                   public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.templatematch";
    std::string nats_url = "nats://127.0.0.1:4222";
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
  void emit_telemetry(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms);
  void set_template_png_b64(const std::string& b64, const json& meta);
  bool ensure_video_open();
  void detect_once();

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;
  std::mutex video_mu_;

  // Template (BGR).
  bool template_loaded_ = false;
  std::string template_error_;
  cv::Mat template_bgr_;
  std::string template_png_b64_;
  double match_threshold_ = 0.50;
  std::int64_t matching_interval_ms_ = 200;
  std::int64_t last_match_ts_ms_ = 0;

  // Video input (BGRA32 SHM).
  std::string shm_name_override_;
  f8::cppsdk::VideoSharedMemoryReader video_;
  std::vector<std::byte> frame_bgra_;
  std::optional<f8::cppsdk::VideoSharedMemoryHeader> last_header_;
  std::uint64_t last_frame_id_ = 0;
  std::uint32_t last_notify_seq_ = 0;
  std::int64_t last_video_open_attempt_ms_ = 0;

  std::uint64_t telemetry_observed_frames_ = 0;
  std::uint64_t telemetry_processed_frames_ = 0;
  std::uint64_t telemetry_window_processed_frames_ = 0;
  std::int64_t telemetry_window_start_ms_ = 0;
  double telemetry_last_process_ms_ = 0.0;
  double telemetry_total_process_ms_ = 0.0;
  double telemetry_fps_ = 0.0;
};

}  // namespace f8::cvkit::template_match
