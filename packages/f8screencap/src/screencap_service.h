#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/shm/video.h"

namespace f8::cppsdk {
class VideoSharedMemorySink;
}

namespace f8::screencap {

class Win32WgcCapture;
class LinuxX11Capture;

#if defined(_WIN32)
using CaptureBackend = Win32WgcCapture;
#else
using CaptureBackend = LinuxX11Capture;
#endif

class ScreenCapService final : public f8::cppsdk::LifecycleNode,
                               public f8::cppsdk::StatefulNode,
                               public f8::cppsdk::SetStateHandlerNode,
                               public f8::cppsdk::RungraphHandlerNode,
                               public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.screencap";
    std::string nats_url = "nats://127.0.0.1:4222";

    std::size_t video_shm_bytes = f8::cppsdk::shm::kDefaultVideoShmBytes;
    std::uint32_t video_shm_slots = f8::cppsdk::shm::kDefaultVideoShmSlots;

    std::string mode = "display";  // display|window|region
    double fps = 30.0;
    int display_id = 0;
    std::string window_id;
    std::string region_csv;  // x,y,w,h
    std::string scale_csv;   // w,h
  };

  explicit ScreenCapService(Config cfg);
  ~ScreenCapService();

  bool start();
  void stop();
  bool running() const {
    return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
  }

  void on_lifecycle(bool active, const nlohmann::json& meta) override;
  void on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value, std::int64_t ts_ms,
                const nlohmann::json& meta) override;
  bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                    const nlohmann::json& meta, std::string& error_code, std::string& error_message) override;
  bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                       std::string& error_message) override;
  bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                  nlohmann::json& result, std::string& error_code, std::string& error_message) override;

  void tick();

  static nlohmann::json describe();

 private:
  using json = nlohmann::json;

  void set_active_local(bool active, const json& meta);

  void publish_static_state();
  void publish_dynamic_state();
  void request_capture_restart();

  Config cfg_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};
  std::atomic<bool> armed_{false};

  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;
  std::shared_ptr<f8::cppsdk::VideoSharedMemorySink> shm_;
  std::unique_ptr<CaptureBackend> capture_;

  mutable std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  std::atomic<bool> capture_restart_{false};
  std::atomic<bool> capture_running_{false};
  std::atomic<std::uint64_t> frame_id_{0};
  std::atomic<std::int64_t> last_frame_ts_ms_{0};
  std::string last_error_;
  std::int64_t last_state_pub_ms_ = 0;

  std::atomic<bool> picker_running_{false};
  std::mutex picker_mu_;
  std::optional<json> picker_pending_patch_;
};

}  // namespace f8::screencap
