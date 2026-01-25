#include "ws_signaling_server.h"

#include <chrono>
#include <utility>
#include <vector>

namespace f8::webrtc_gateway {

WsSignalingServer::WsSignalingServer() = default;

WsSignalingServer::~WsSignalingServer() {
  stop();
}

bool WsSignalingServer::start(const Config& cfg, MessageCallback on_message, ConnectionCallback on_connect,
                              ConnectionCallback on_disconnect, std::string& err) {
  stop();
  cfg_ = cfg;
  on_message_ = std::move(on_message);
  on_connect_ = std::move(on_connect);
  on_disconnect_ = std::move(on_disconnect);

  try {
    server_.clear_access_channels(websocketpp::log::alevel::all);
    server_.clear_error_channels(websocketpp::log::elevel::all);
    server_.init_asio();
    server_.set_reuse_addr(true);

    server_.set_open_handler([this](connection_hdl hdl) {
      std::string id = "c" + std::to_string(seq_.fetch_add(1, std::memory_order_relaxed) + 1);
      std::size_t cnt = 0;
      {
        std::lock_guard<std::mutex> lock(mu_);
        clients_by_id_[id] = hdl;
        ids_by_hdl_[hdl] = id;
        cnt = clients_by_id_.size();
      }
      connection_count_.store(cnt, std::memory_order_relaxed);
      if (on_connect_)
        on_connect_(id, cnt);
    });

    server_.set_close_handler([this](connection_hdl hdl) {
      std::string id;
      std::size_t cnt = 0;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = ids_by_hdl_.find(hdl);
        if (it != ids_by_hdl_.end()) {
          id = it->second;
          ids_by_hdl_.erase(it);
          clients_by_id_.erase(id);
        }
        cnt = clients_by_id_.size();
      }
      connection_count_.store(cnt, std::memory_order_relaxed);
      if (!id.empty() && on_disconnect_)
        on_disconnect_(id, cnt);
    });

    server_.set_message_handler([this](connection_hdl hdl, Server::message_ptr msg) {
      std::string id;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = ids_by_hdl_.find(hdl);
        if (it != ids_by_hdl_.end())
          id = it->second;
      }
      if (id.empty() || !msg)
        return;
      if (on_message_) {
        Event ev;
        ev.kind = Event::Kind::Message;
        ev.client_id = id;
        ev.text = msg->get_payload();
        on_message_(ev);
      }
    });

    websocketpp::lib::asio::ip::address addr;
    try {
      addr = websocketpp::lib::asio::ip::make_address(cfg_.host);
    } catch (...) {
      err = "invalid host";
      return false;
    }
    websocketpp::lib::error_code ec;
    websocketpp::lib::asio::ip::tcp::endpoint ep(addr, cfg_.port);
    server_.listen(ep, ec);
    if (ec) {
      err = ec.message();
      return false;
    }
    server_.start_accept();

    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this]() { run(); });
    return true;
  } catch (const std::exception& e) {
    err = e.what();
    stop();
    return false;
  } catch (...) {
    err = "unknown error";
    stop();
    return false;
  }
}

void WsSignalingServer::run() {
  try {
    server_.run();
  } catch (...) {}
}

void WsSignalingServer::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  try {
    server_.stop_listening();
  } catch (...) {}
  try {
    server_.stop();
  } catch (...) {}
  try {
    if (thread_.joinable())
      thread_.join();
  } catch (...) {}
  {
    std::lock_guard<std::mutex> lock(mu_);
    clients_by_id_.clear();
    ids_by_hdl_.clear();
  }
  connection_count_.store(0, std::memory_order_relaxed);
}

bool WsSignalingServer::sendText(const std::string& client_id, const std::string& text) {
  connection_hdl hdl;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = clients_by_id_.find(client_id);
    if (it == clients_by_id_.end())
      return false;
    hdl = it->second;
  }

  try {
    const std::string payload = text;
    server_.get_io_service().post([this, hdl, payload]() {
      websocketpp::lib::error_code ec;
      server_.send(hdl, payload, websocketpp::frame::opcode::text, ec);
    });
    return true;
  } catch (...) {
    return false;
  }
}

void WsSignalingServer::broadcastText(const std::string& text) {
  std::vector<connection_hdl> targets;
  {
    std::lock_guard<std::mutex> lock(mu_);
    targets.reserve(clients_by_id_.size());
    for (const auto& kv : clients_by_id_) {
      targets.push_back(kv.second);
    }
  }

  const std::string payload = text;
  try {
    server_.get_io_service().post([this, targets, payload]() {
      for (const auto& hdl : targets) {
        websocketpp::lib::error_code ec;
        server_.send(hdl, payload, websocketpp::frame::opcode::text, ec);
      }
    });
  } catch (...) {}
}

}  // namespace f8::webrtc_gateway
