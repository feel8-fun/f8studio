#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/video_shared_memory_sink.h"
#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/nats_client.h"
#include "f8cppsdk/service_control_plane.h"
#include "f8cppsdk/service_control_plane_server.h"
#include "ws_signaling_server.h"

namespace f8::webrtc_gateway {

class WebRtcGatewayService final : public f8::cppsdk::ServiceControlHandler {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.webrtc.gateway";
    std::string nats_url = "nats://127.0.0.1:4222";
    std::uint16_t ws_port = 8765;
    bool video_force_h264 = false;
    bool video_use_gstreamer = false;
  };

  explicit WebRtcGatewayService(Config cfg);
  ~WebRtcGatewayService();

  bool start();
  void stop();
  bool running() const {
    return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
  }

  void tick();

  // ServiceControlHandler
  bool is_active() const override { return active_.load(std::memory_order_acquire); }
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
  struct WebRtcSession;
  struct WsOutbound {
    std::string client_id;
    std::string text;
  };

  bool restart_ws(std::string& err);
  void set_active_local(bool active, const nlohmann::json& meta);
  void publish_static_state();
  void publish_dynamic_state();
  void process_video(std::int64_t now_ms);
  void maybe_request_periodic_keyframes(std::int64_t now_ms);

  void enqueue_ws_event(const WsSignalingServer::Event& ev);
  std::vector<WsSignalingServer::Event> drain_ws_events();
  void enqueue_ws_send(std::string client_id, std::string text);
  std::vector<WsOutbound> drain_ws_sends();

  void handle_ws_event(const WsSignalingServer::Event& ev);
  void handle_ws_json_message(const WsSignalingServer::Event& ev, const nlohmann::json& msg);
  void stop_session_by_id(const std::string& session_id, const std::string& reason);
  void stop_sessions_by_client(const std::string& client_id, const std::string& reason);

  // Optional GStreamer webrtcbin receive+decode path.
  bool handle_offer_gst(const std::string& client_id, const std::string& session_id, const std::string& offer_sdp);
  bool handle_ice_gst(const std::string& session_id, int mline, const std::string& candidate);
  void stop_session_gst(WebRtcSession& session);
  void tick_gst();

  static void gst_on_pad_added(void* webrtc, void* pad, void* user_data);
  static void gst_on_ice_candidate(void* webrtc, unsigned mline, char* candidate, void* user_data);
  static int gst_on_new_sample(void* appsink, void* user_data);

  Config cfg_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  f8::cppsdk::NatsClient nats_;
  f8::cppsdk::KvStore kv_;
  std::unique_ptr<f8::cppsdk::ServiceControlPlaneServer> ctrl_;

  WsSignalingServer ws_;

  mutable std::mutex state_mu_;
  std::unordered_map<std::string, nlohmann::json> published_state_;
  std::int64_t last_state_pub_ms_ = 0;

  // Decoded video output (BGRA32) to SHM for downstream analysis/preview.
  mutable std::mutex video_mu_;
  f8::cppsdk::VideoSharedMemorySink video_shm_;
  std::string video_shm_name_;
  std::size_t video_shm_bytes_ = 256ull * 1024ull * 1024ull;
  std::uint32_t video_shm_slots_ = 2;
  std::int64_t last_video_write_ms_ = 0;
  int video_shm_max_width_ = 1920;
  int video_shm_max_height_ = 1080;
  int video_shm_max_fps_ = 30;
  bool video_enabled_ = true;
  bool video_force_h264_ = false;
  bool video_use_gstreamer_ = false;
  std::atomic<std::uint64_t> video_frames_rx_{0};
  std::atomic<std::uint64_t> video_frames_decoded_{0};
  std::atomic<std::uint64_t> video_frames_written_{0};
  std::atomic<std::uint64_t> video_decode_errors_{0};
  std::atomic<std::uint64_t> video_unsupported_tracks_{0};
  std::atomic<std::uint64_t> video_rtp_packets_rx_{0};
  std::atomic<std::uint64_t> video_rtp_bytes_rx_{0};
  std::atomic<int> video_last_rtp_pt_{-1};
  std::atomic<std::uint64_t> video_last_frame_bytes_{0};
  std::atomic<std::int64_t> video_last_frame_ts_ms_{0};
  std::atomic<std::int64_t> video_last_decode_ms_{0};
  mutable std::mutex video_err_mu_;
  std::string video_last_error_;

  std::atomic<std::uint64_t> ice_tx_{0};
  std::atomic<std::uint64_t> ice_rx_{0};
  mutable std::mutex ice_err_mu_;
  std::string ice_last_error_;

  mutable std::mutex ws_mu_;
  std::vector<WsSignalingServer::Event> ws_events_;

  mutable std::mutex ws_out_mu_;
  std::vector<WsOutbound> ws_out_;

  mutable std::mutex rtc_mu_;
  std::unordered_map<std::string, std::unique_ptr<WebRtcSession>> sessions_by_id_;
};

}  // namespace f8::webrtc_gateway
