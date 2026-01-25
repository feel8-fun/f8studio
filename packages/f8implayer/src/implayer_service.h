#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/nats_client.h"
#include "f8cppsdk/service_control_plane.h"
#include "f8cppsdk/service_control_plane_server.h"

namespace f8::implayer {

class MpvPlayer;
class SdlVideoWindow;
class ImPlayerGui;
class VideoSharedMemorySink;

class ImPlayerService final : public f8::cppsdk::ServiceControlHandler {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.implayer";
    std::string nats_url = "nats://127.0.0.1:4222";

    std::size_t video_shm_bytes = 256ull * 1024ull * 1024ull;
    std::uint32_t video_shm_slots = 2;
    std::uint32_t video_shm_max_width = 1920;
    std::uint32_t video_shm_max_height = 1080;
    double video_shm_max_fps = 30.0;

    int window_width = 1280;
    int window_height = 720;
    bool window_resizable = true;
    bool window_vsync = true;

    std::string initial_media_url;
  };

  explicit ImPlayerService(Config cfg);
  ~ImPlayerService();

  bool start();
  void stop();
  bool running() const {
    return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
  }

  // Called on the main thread periodically.
  void tick();

  // ServiceControlHandler
  void on_activate(const nlohmann::json& meta) override;
  void on_deactivate(const nlohmann::json& meta) override;
  void on_set_active(bool active, const nlohmann::json& meta) override;
  bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                    const nlohmann::json& meta, std::string& error_code, std::string& error_message) override;
  bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                       std::string& error_message) override;
  bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                  nlohmann::json& result, std::string& error_code, std::string& error_message) override;

  static nlohmann::json describe();

 private:
  void set_active_local(bool active, const nlohmann::json& meta);
  void publish_static_state();
  void publish_dynamic_state();

  bool cmd_open(const nlohmann::json& args, std::string& err);
  bool cmd_play(std::string& err);
  bool cmd_pause(std::string& err);
  bool cmd_stop(std::string& err);
  bool cmd_seek(const nlohmann::json& args, std::string& err);
  bool cmd_set_volume(const nlohmann::json& args, std::string& err);

  Config cfg_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  f8::cppsdk::NatsClient nats_;
  f8::cppsdk::KvStore kv_;
  std::unique_ptr<f8::cppsdk::ServiceControlPlaneServer> ctrl_;

  std::unique_ptr<SdlVideoWindow> window_;
  std::unique_ptr<ImPlayerGui> gui_;
  std::unique_ptr<MpvPlayer> player_;
  std::shared_ptr<VideoSharedMemorySink> shm_;

  mutable std::mutex state_mu_;
  std::string media_url_;
  double volume_ = 1.0;
  std::string last_error_;

  std::atomic<double> position_seconds_{0.0};
  std::atomic<double> duration_seconds_{0.0};
  std::atomic<bool> playing_{false};

  std::int64_t last_state_pub_ms_ = 0;
};

}  // namespace f8::implayer
