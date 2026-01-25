#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

namespace f8::webrtc_gateway {

class WsSignalingServer {
 public:
  struct Config {
    std::string host = "127.0.0.1";
    std::uint16_t port = 8765;
  };

  struct Event {
    enum class Kind { Message, Connect, Disconnect };

    Kind kind = Kind::Message;
    std::string client_id;
    std::string text;
  };

  using MessageCallback = std::function<void(const Event&)>;
  using ConnectionCallback = std::function<void(const std::string& client_id, std::size_t connections)>;

  WsSignalingServer();
  ~WsSignalingServer();

  WsSignalingServer(const WsSignalingServer&) = delete;
  WsSignalingServer& operator=(const WsSignalingServer&) = delete;

  bool start(const Config& cfg, MessageCallback on_message, ConnectionCallback on_connect,
             ConnectionCallback on_disconnect, std::string& err);
  void stop();

  std::size_t connectionCount() const { return connection_count_.load(std::memory_order_relaxed); }

  bool sendText(const std::string& client_id, const std::string& text);
  void broadcastText(const std::string& text);

 private:
  using Server = websocketpp::server<websocketpp::config::asio>;
  using connection_hdl = websocketpp::connection_hdl;

  void run();

  mutable std::mutex mu_;
  Server server_;
  Config cfg_{};
  MessageCallback on_message_;
  ConnectionCallback on_connect_;
  ConnectionCallback on_disconnect_;
  std::thread thread_;

  std::unordered_map<std::string, connection_hdl> clients_by_id_;
  std::map<connection_hdl, std::string, std::owner_less<connection_hdl>> ids_by_hdl_;
  std::atomic<std::size_t> connection_count_{0};
  std::atomic<std::uint64_t> seq_{0};
  std::atomic<bool> running_{false};
};

}  // namespace f8::webrtc_gateway
