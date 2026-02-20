#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/video_shared_memory_sink.h"

namespace f8::cvkit::tracking {

struct TrackingInitCandidate {
  cv::Rect bbox;
  std::optional<double> score;
};

enum class TrackingInitSelectMode {
  ClosestCenter,
  LargestArea,
  HighestScore,
};

class TrackingService final : public f8::cppsdk::LifecycleNode,
                              public f8::cppsdk::StatefulNode,
                              public f8::cppsdk::DataReceivableNode,
                              public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.tracking";
    std::string nats_url = "nats://127.0.0.1:4222";
    std::string shm_name;
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
  void set_shm_name(const std::string& shm_name, const json& meta);
  void set_init_select(const std::string& mode, const json& meta);
  bool ensure_video_open();
   void apply_init_box_if_any();
   void process_frame_once();
   void set_tracking(bool tracking, const json& meta);
   void stop_tracking_internal(const json& meta);

   Config cfg_;
   std::atomic<bool> running_{false};
   std::atomic<bool> stop_requested_{false};
   std::atomic<bool> active_{true};
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  // Video input (BGRA32 SHM).
  std::string shm_name_override_;
  f8::cppsdk::VideoSharedMemoryReader video_;
  std::vector<std::byte> frame_bgra_;
  std::optional<f8::cppsdk::VideoSharedMemoryHeader> last_header_;
  std::uint64_t last_frame_id_ = 0;
  std::int64_t last_video_open_attempt_ms_ = 0;
  TrackingInitSelectMode init_select_mode_ = TrackingInitSelectMode::ClosestCenter;
  std::string init_select_state_ = "closest_center";

   // Tracking state.
   std::mutex tracking_mu_;
   cv::Ptr<cv::Tracker> tracker_;
   cv::Rect bbox_;
   bool is_tracking_ = false;

  // Pending init candidates extracted from upstream payloads.
  std::vector<TrackingInitCandidate> pending_init_boxes_;
};

}  // namespace f8::cvkit::tracking
