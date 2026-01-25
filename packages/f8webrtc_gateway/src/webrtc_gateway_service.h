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

  void enqueue_ws_event(const WsSignalingServer::Event& ev);
  std::vector<WsSignalingServer::Event> drain_ws_events();
  void enqueue_ws_send(std::string client_id, std::string text);
  std::vector<WsOutbound> drain_ws_sends();

  void handle_ws_event(const WsSignalingServer::Event& ev);
  void handle_ws_json_message(const WsSignalingServer::Event& ev, const nlohmann::json& msg);
  void stop_session_by_id(const std::string& session_id, const std::string& reason);
  void stop_sessions_by_client(const std::string& client_id, const std::string& reason);

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

  mutable std::mutex ws_mu_;
  std::vector<WsSignalingServer::Event> ws_events_;

  mutable std::mutex ws_out_mu_;
  std::vector<WsOutbound> ws_out_;

  mutable std::mutex rtc_mu_;
  std::unordered_map<std::string, std::unique_ptr<WebRtcSession>> sessions_by_id_;
};

}  // namespace f8::webrtc_gateway
