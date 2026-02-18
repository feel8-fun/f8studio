#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <SDL3/SDL.h>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/shm/video.h"
#include "sdl_video_window.h"

namespace f8::cppsdk {
class VideoSharedMemorySink;
}

namespace f8::implayer {

class MpvPlayer;
class ImPlayerGui;
using VideoSharedMemorySink = ::f8::cppsdk::VideoSharedMemorySink;

class ImPlayerService final : public f8::cppsdk::LifecycleNode,
                              public f8::cppsdk::StatefulNode,
                              public f8::cppsdk::SetStateHandlerNode,
                              public f8::cppsdk::RungraphHandlerNode,
                              public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.implayer";
    std::string nats_url = "nats://127.0.0.1:4222";

    std::size_t video_shm_bytes = f8::cppsdk::shm::kDefaultVideoShmBytes;
    std::uint32_t video_shm_slots = f8::cppsdk::shm::kDefaultVideoShmSlots;
    std::uint32_t video_shm_max_width = 600;
    std::uint32_t video_shm_max_height = 600;
    double video_shm_max_fps = 40.0;

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

  void on_lifecycle(bool active, const nlohmann::json& meta) override;
  void on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value, std::int64_t ts_ms,
                const nlohmann::json& meta) override;

  // When using SDL_MAIN_USE_CALLBACKS, feed events here from SDL_AppEvent.
  void processSdlEvent(const SDL_Event& ev);

  // Called on the main thread periodically.
  void tick();

  static nlohmann::json describe();

 private:
  void set_active_local(bool active);
  bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                    const nlohmann::json& meta, std::string& error_code, std::string& error_message) override;
  bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                       std::string& error_message) override;
  bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                  nlohmann::json& result, std::string& error_code, std::string& error_message) override;
  void publish_static_state();
  void publish_dynamic_state();

  void playlist_add(const std::vector<std::string>& items, bool play_if_idle);
  void playlist_play_index(int index);
  void playlist_remove_index(int index);
  void playlist_clear();
  void playlist_next();
  void playlist_prev();

  bool open_media_internal(const std::string& url, bool keep_playlist, std::string& err);

  bool cmd_open(const nlohmann::json& args, std::string& err);
  bool cmd_play(std::string& err);
  bool cmd_pause(std::string& err);
  bool cmd_stop(std::string& err);
  bool cmd_seek(const nlohmann::json& args, std::string& err);
  bool cmd_set_volume(const nlohmann::json& args, std::string& err);
  void mark_vr_manual_override();

  Config cfg_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::unique_ptr<SdlVideoWindow> window_;
  std::unique_ptr<ImPlayerGui> gui_;
  std::unique_ptr<MpvPlayer> player_;
  std::shared_ptr<VideoSharedMemorySink> shm_;

  mutable std::mutex state_mu_;
  std::string media_url_;
  double volume_ = 1.0;
  std::string last_error_;
  std::string video_id_;
  std::unordered_map<std::string, nlohmann::json> published_state_;

  std::atomic<double> position_seconds_{0.0};
  std::atomic<double> duration_seconds_{0.0};
  std::atomic<bool> playing_{false};
  std::atomic<bool> media_finished_{false};
  std::atomic<bool> eof_reached_{false};
  std::atomic<bool> stopped_{false};
  std::atomic<bool> clear_video_requested_{false};

  std::int64_t last_state_pub_ms_ = 0;
  std::int64_t last_playback_data_pub_ms_ = 0;
  std::uint64_t last_frame_id_published_ = 0;
  std::int64_t last_tick_ms_ = 0;
  double tick_ema_ms_ = 0.0;
  double tick_ema_fps_ = 0.0;

  std::vector<std::string> playlist_;
  int playlist_index_ = -1;

  // View transform (window-local, not published).
  float view_zoom_ = 1.0f;
  float view_pan_x_ = 0.0f;
  float view_pan_y_ = 0.0f;
  bool view_panning_ = false;
  float view_pan_anchor_x_ = 0.0f;
  float view_pan_anchor_y_ = 0.0f;
  float view_pan_start_x_ = 0.0f;
  float view_pan_start_y_ = 0.0f;
  unsigned view_last_video_w_ = 0;
  unsigned view_last_video_h_ = 0;

  bool loop_ = false;

  SdlVideoWindow::ProjectionMode vr_mode_ = SdlVideoWindow::ProjectionMode::Flat2D;
  float vr_yaw_deg_ = 0.0f;
  float vr_pitch_deg_ = 0.0f;
  float vr_fov_deg_ = 90.0f;
  int vr_sbs_eye_ = 0;

  bool vr_dragging_ = false;
  float vr_drag_anchor_x_ = 0.0f;
  float vr_drag_anchor_y_ = 0.0f;
  float vr_drag_start_yaw_deg_ = 0.0f;
  float vr_drag_start_pitch_deg_ = 0.0f;

  SdlVideoWindow::ProjectionMode vr_auto_detect_mode_ = SdlVideoWindow::ProjectionMode::Flat2D;
  bool vr_auto_detect_valid_ = false;
  bool vr_auto_pending_ratio_ = false;
  bool vr_manual_override_ = false;
  std::string vr_auto_video_id_;

  std::mutex render_mu_;
};

}  // namespace f8::implayer
