#include "webrtc_gateway_service.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <rtc/rtc.hpp>

#include "f8cppsdk/data_bus.h"
#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"

namespace f8::webrtc_gateway {

using json = nlohmann::json;

namespace {

json schema_string() {
  return json{{"type", "string"}};
}
json schema_integer() {
  return json{{"type", "integer"}};
}
json schema_boolean() {
  return json{{"type", "boolean"}};
}
json schema_object(const json& props, const json& required = json::array()) {
  json obj;
  obj["type"] = "object";
  obj["properties"] = props;
  if (required.is_array())
    obj["required"] = required;
  obj["additionalProperties"] = false;
  return obj;
}

json state_field(std::string name, const json& value_schema, std::string access, std::string label = {},
                 std::string description = {}, bool show_on_node = false) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  if (!label.empty())
    sf["label"] = std::move(label);
  if (!description.empty())
    sf["description"] = std::move(description);
  if (show_on_node)
    sf["showOnNode"] = true;
  return sf;
}

std::string ws_url(std::uint16_t port) {
  return "ws://127.0.0.1:" + std::to_string(static_cast<unsigned>(port)) + "/ws";
}

rtc::LogLevel rtc_level() {
  return rtc::LogLevel::Warning;
}

std::optional<std::string> json_string(const json& obj, const char* key) {
  if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string())
    return std::nullopt;
  return obj[key].get<std::string>();
}

std::optional<json> json_object(const json& obj, const char* key) {
  if (!obj.is_object() || !obj.contains(key) || !obj[key].is_object())
    return std::nullopt;
  return obj[key];
}

}  // namespace

struct WebRtcGatewayService::WebRtcSession {
  std::string session_id;
  std::string client_id;
  std::unique_ptr<rtc::PeerConnection> pc;
  std::shared_ptr<rtc::DataChannel> dc;
  bool answer_sent = false;
  std::int64_t created_ms = 0;
  std::int64_t last_msg_ms = 0;
};

WebRtcGatewayService::WebRtcGatewayService(Config cfg) : cfg_(std::move(cfg)) {}

WebRtcGatewayService::~WebRtcGatewayService() {
  stop();
}

bool WebRtcGatewayService::restart_ws(std::string& err) {
  ws_.stop();

  WsSignalingServer::Config wcfg;
  wcfg.host = "127.0.0.1";
  wcfg.port = cfg_.ws_port;

  auto on_msg = [this](const WsSignalingServer::Event& ev) {
    enqueue_ws_event(ev);
  };
  auto on_connect = [this](const std::string& client_id, std::size_t) {
    WsSignalingServer::Event ev;
    ev.kind = WsSignalingServer::Event::Kind::Connect;
    ev.client_id = client_id;
    enqueue_ws_event(ev);
  };
  auto on_disconnect = [this](const std::string& client_id, std::size_t) {
    WsSignalingServer::Event ev;
    ev.kind = WsSignalingServer::Event::Kind::Disconnect;
    ev.client_id = client_id;
    enqueue_ws_event(ev);
  };

  return ws_.start(wcfg, std::move(on_msg), std::move(on_connect), std::move(on_disconnect), err);
}

bool WebRtcGatewayService::start() {
  if (running_.load(std::memory_order_acquire))
    return true;

  try {
    rtc::InitLogger(rtc_level(), [](rtc::LogLevel, rtc::string message) { spdlog::debug("rtc: {}", message); });
  } catch (...) {}

  try {
    cfg_.service_id = f8::cppsdk::ensure_token(cfg_.service_id, "service_id");
  } catch (const std::exception& e) {
    spdlog::error("invalid --service-id: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("invalid --service-id");
    return false;
  }

  if (!nats_.connect(cfg_.nats_url))
    return false;

  f8::cppsdk::KvConfig kvc;
  kvc.bucket = f8::cppsdk::kv_bucket_for_service(cfg_.service_id);
  kvc.history = 1;
  kvc.memory_storage = true;
  if (!kv_.open_or_create(nats_.jetstream(), kvc))
    return false;

  ctrl_ = std::make_unique<f8::cppsdk::ServiceControlPlaneServer>(
      f8::cppsdk::ServiceControlPlaneServer::Config{cfg_.service_id, cfg_.nats_url}, &nats_, &kv_, this);
  if (!ctrl_->start()) {
    spdlog::error("failed to start control plane");
    return false;
  }

  std::string err;
  if (!restart_ws(err)) {
    spdlog::error("failed to start websocket server: {}", err);
    return false;
  }

  publish_static_state();
  publish_dynamic_state();
  f8::cppsdk::kv_set_ready(kv_, true);

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("webrtc_gateway started serviceId={} wsPort={}", cfg_.service_id, static_cast<unsigned>(cfg_.ws_port));
  return true;
}

void WebRtcGatewayService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  stop_requested_.store(true, std::memory_order_release);

  try {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    for (auto& kv : sessions_by_id_) {
      if (kv.second && kv.second->pc)
        kv.second->pc->close();
    }
    sessions_by_id_.clear();
  } catch (...) {}

  try {
    ws_.stop();
  } catch (...) {}

  try {
    if (ctrl_)
      ctrl_->stop();
  } catch (...) {}
  ctrl_.reset();

  kv_.stop_watch();
  kv_.close();
  nats_.close();

  try {
    rtc::Cleanup();
  } catch (...) {}
}

void WebRtcGatewayService::enqueue_ws_event(const WsSignalingServer::Event& ev) {
  std::lock_guard<std::mutex> lock(ws_mu_);
  ws_events_.push_back(ev);
}

std::vector<WsSignalingServer::Event> WebRtcGatewayService::drain_ws_events() {
  std::vector<WsSignalingServer::Event> out;
  std::lock_guard<std::mutex> lock(ws_mu_);
  out.swap(ws_events_);
  return out;
}

void WebRtcGatewayService::enqueue_ws_send(std::string client_id, std::string text) {
  if (client_id.empty() || text.empty())
    return;
  std::lock_guard<std::mutex> lock(ws_out_mu_);
  ws_out_.push_back(WsOutbound{std::move(client_id), std::move(text)});
}

std::vector<WebRtcGatewayService::WsOutbound> WebRtcGatewayService::drain_ws_sends() {
  std::vector<WsOutbound> out;
  std::lock_guard<std::mutex> lock(ws_out_mu_);
  out.swap(ws_out_);
  return out;
}

void WebRtcGatewayService::tick() {
  if (!running_.load(std::memory_order_acquire))
    return;

  const auto events = drain_ws_events();
  if (!events.empty()) {
    const std::int64_t now = f8::cppsdk::now_ms();
    for (const auto& ev : events) {
      if (!active_.load(std::memory_order_acquire))
        continue;
      handle_ws_event(ev);
      if (ev.kind != WsSignalingServer::Event::Kind::Message || ev.text.empty())
        continue;

      json payload;
      payload["clientId"] = ev.client_id;
      payload["text"] = ev.text;
      payload["wsUrl"] = ws_url(cfg_.ws_port);
      payload["ts"] = now;

      // Try to parse as JSON so downstream can route by type/payload.
      try {
        json parsed = json::parse(ev.text, nullptr, false);
        if (parsed.is_object() || parsed.is_array()) {
          payload["json"] = parsed;
        }
      } catch (...) {}

      (void)f8::cppsdk::publish_data(nats_, cfg_.service_id, cfg_.service_id, "signalRx", payload, now);
    }
  }

  const auto out = drain_ws_sends();
  for (const auto& msg : out) {
    ws_.sendText(msg.client_id, msg.text);
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (now - last_state_pub_ms_ >= 200) {
    publish_dynamic_state();
    last_state_pub_ms_ = now;
  }
}

void WebRtcGatewayService::handle_ws_event(const WsSignalingServer::Event& ev) {
  if (ev.kind == WsSignalingServer::Event::Kind::Connect) {
    spdlog::info("ws connect clientId={} connections={}", ev.client_id, ws_.connectionCount());
    return;
  }
  if (ev.kind == WsSignalingServer::Event::Kind::Disconnect) {
    spdlog::info("ws disconnect clientId={} connections={}", ev.client_id, ws_.connectionCount());
    stop_sessions_by_client(ev.client_id, "ws_disconnect");
    return;
  }

  if (ev.text.empty())
    return;
  try {
    json parsed = json::parse(ev.text, nullptr, false);
    if (parsed.is_object())
      handle_ws_json_message(ev, parsed);
  } catch (...) {}
}

void WebRtcGatewayService::handle_ws_json_message(const WsSignalingServer::Event& ev, const nlohmann::json& msg) {
  const auto type = json_string(msg, "type").value_or("");
  if (type.empty())
    return;

  if (type == "hello") {
    spdlog::info("ws hello clientId={} msg={}", ev.client_id, msg.dump());
    return;
  }

  if (type == "webrtc.stop") {
    const auto sid = json_string(msg, "sessionId").value_or("");
    if (!sid.empty())
      stop_session_by_id(sid, json_string(msg, "reason").value_or("stopped"));
    return;
  }

  if (type == "webrtc.ice") {
    const auto sid = json_string(msg, "sessionId").value_or("");
    if (sid.empty())
      return;
    const auto candObj = json_object(msg, "candidate");
    if (!candObj)
      return;
    const auto candStr = json_string(*candObj, "candidate").value_or("");
    if (candStr.empty())
      return;
    const auto mid = json_string(*candObj, "sdpMid").value_or("");

    rtc::PeerConnection* pc = nullptr;
    {
      std::lock_guard<std::mutex> lock(rtc_mu_);
      auto it = sessions_by_id_.find(sid);
      if (it == sessions_by_id_.end() || !it->second || !it->second->pc)
        return;
      pc = it->second->pc.get();
      it->second->last_msg_ms = f8::cppsdk::now_ms();
    }

    try {
      rtc::Candidate c(mid.empty() ? rtc::Candidate(candStr) : rtc::Candidate(candStr, mid));
      pc->addRemoteCandidate(c);
      spdlog::debug("webrtc ice rx clientId={} sessionId={} mid={} cand={}", ev.client_id, sid, mid, candStr);
    } catch (const std::exception& e) {
      spdlog::warn("webrtc ice rx failed sessionId={} err={}", sid, e.what());
    } catch (...) {
      spdlog::warn("webrtc ice rx failed sessionId={}", sid);
    }
    return;
  }

  if (type == "webrtc.offer") {
    const auto sid = json_string(msg, "sessionId").value_or("");
    const auto descObj = json_object(msg, "description");
    if (sid.empty() || !descObj)
      return;

    const auto sdpType = json_string(*descObj, "type").value_or("offer");
    const auto sdp = json_string(*descObj, "sdp").value_or("");
    if (sdp.empty())
      return;

    spdlog::info("webrtc offer rx clientId={} sessionId={} sdpBytes={}", ev.client_id, sid, sdp.size());

    auto session = std::make_unique<WebRtcSession>();
    session->session_id = sid;
    session->client_id = ev.client_id;
    session->created_ms = f8::cppsdk::now_ms();
    session->last_msg_ms = session->created_ms;

    rtc::Configuration cfg;
    session->pc = std::make_unique<rtc::PeerConnection>(cfg);
    rtc::PeerConnection* const pc = session->pc.get();

    const std::string clientId = ev.client_id;
    const std::string sessionId = sid;

    session->pc->onStateChange([clientId, sessionId](rtc::PeerConnection::State state) {
      spdlog::info("webrtc pc state clientId={} sessionId={} state={}", clientId, sessionId, static_cast<int>(state));
    });
    session->pc->onIceStateChange([clientId, sessionId](rtc::PeerConnection::IceState state) {
      spdlog::info("webrtc ice state clientId={} sessionId={} state={}", clientId, sessionId, static_cast<int>(state));
    });
    session->pc->onGatheringStateChange([clientId, sessionId](rtc::PeerConnection::GatheringState state) {
      spdlog::debug("webrtc gather state clientId={} sessionId={} state={}", clientId, sessionId,
                    static_cast<int>(state));
    });

    session->pc->onLocalDescription([this, clientId, sessionId](rtc::Description desc) {
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second) {
          if (it->second->answer_sent)
            return;
          it->second->answer_sent = true;
        }
      }
      json out;
      out["type"] = "webrtc.answer";
      out["sessionId"] = sessionId;
      out["description"] = json{{"type", desc.typeString()}, {"sdp", desc.generateSdp()}};
      out["ts"] = f8::cppsdk::now_ms();
      enqueue_ws_send(clientId, out.dump());
      spdlog::info("webrtc answer tx clientId={} sessionId={} sdpBytes={}", clientId, sessionId,
                   out["description"]["sdp"].get<std::string>().size());
    });

    session->pc->onLocalCandidate([this, clientId, sessionId](rtc::Candidate cand) {
      json out;
      out["type"] = "webrtc.ice";
      out["sessionId"] = sessionId;
      out["candidate"] = json{{"candidate", cand.candidate()}, {"sdpMid", cand.mid()}, {"sdpMLineIndex", 0}};
      out["ts"] = f8::cppsdk::now_ms();
      enqueue_ws_send(clientId, out.dump());
    });

    session->pc->onDataChannel([this, clientId, sessionId](std::shared_ptr<rtc::DataChannel> dc) {
      if (!dc)
        return;
      spdlog::info("webrtc datachannel rx clientId={} sessionId={} label={}", clientId, sessionId, dc->label());
      {
        std::lock_guard<std::mutex> lock(rtc_mu_);
        auto it = sessions_by_id_.find(sessionId);
        if (it != sessions_by_id_.end() && it->second)
          it->second->dc = dc;
      }
      dc->onOpen([this, clientId, sessionId]() {
        spdlog::info("webrtc dc open clientId={} sessionId={}", clientId, sessionId);
        json out;
        out["type"] = "webrtc.debug";
        out["sessionId"] = sessionId;
        out["event"] = "dcOpen";
        out["ts"] = f8::cppsdk::now_ms();
        enqueue_ws_send(clientId, out.dump());
      });
      dc->onClosed([this, clientId, sessionId]() {
        spdlog::info("webrtc dc closed clientId={} sessionId={}", clientId, sessionId);
        json out;
        out["type"] = "webrtc.debug";
        out["sessionId"] = sessionId;
        out["event"] = "dcClosed";
        out["ts"] = f8::cppsdk::now_ms();
        enqueue_ws_send(clientId, out.dump());
      });
      dc->onMessage([this, clientId, sessionId, dc](rtc::message_variant data) {
        if (const auto* text = std::get_if<std::string>(&data)) {
          spdlog::info("webrtc dc msg rx clientId={} sessionId={} bytes={}", clientId, sessionId, text->size());
          json echo;
          echo["type"] = "webrtc.debug";
          echo["sessionId"] = sessionId;
          echo["event"] = "dcMessage";
          echo["text"] = *text;
          echo["ts"] = f8::cppsdk::now_ms();
          enqueue_ws_send(clientId, echo.dump());
          try {
            dc->send(*text);
          } catch (...) {}
        }
      });
    });

    session->pc->onTrack([clientId, sessionId](std::shared_ptr<rtc::Track> tr) {
      if (!tr)
        return;
      try {
        const auto desc = tr->description();
        spdlog::info("webrtc track rx clientId={} sessionId={} mid={} type={} dir={}", clientId, sessionId, tr->mid(),
                     desc.type(), static_cast<int>(desc.direction()));
      } catch (...) {
        spdlog::info("webrtc track rx clientId={} sessionId={} mid={}", clientId, sessionId, tr->mid());
      }
    });

    {
      std::lock_guard<std::mutex> lock(rtc_mu_);
      sessions_by_id_[sid] = std::move(session);
    }

    try {
      pc->setRemoteDescription(rtc::Description(sdp, sdpType));
      pc->setLocalDescription();
      pc->gatherLocalCandidates();
    } catch (const std::exception& e) {
      spdlog::error("webrtc offer handle failed sessionId={} err={}", sid, e.what());
      stop_session_by_id(sid, "offer_failed");
      return;
    } catch (...) {
      spdlog::error("webrtc offer handle failed sessionId={}", sid);
      stop_session_by_id(sid, "offer_failed");
      return;
    }

    json ack;
    ack["type"] = "webrtc.accepted";
    ack["sessionId"] = sid;
    ack["ts"] = f8::cppsdk::now_ms();
    enqueue_ws_send(ev.client_id, ack.dump());
    return;
  }
}

void WebRtcGatewayService::stop_session_by_id(const std::string& session_id, const std::string& reason) {
  std::unique_ptr<WebRtcSession> sess;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = sessions_by_id_.find(session_id);
    if (it == sessions_by_id_.end())
      return;
    sess = std::move(it->second);
    sessions_by_id_.erase(it);
  }

  if (sess && sess->pc) {
    spdlog::info("webrtc stop sessionId={} reason={}", session_id, reason);
    try {
      sess->pc->close();
    } catch (...) {}
  }
}

void WebRtcGatewayService::stop_sessions_by_client(const std::string& client_id, const std::string& reason) {
  if (client_id.empty())
    return;
  std::vector<std::string> to_stop;
  {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    for (const auto& kv : sessions_by_id_) {
      if (kv.second && kv.second->client_id == client_id)
        to_stop.push_back(kv.first);
    }
  }
  for (const auto& sid : to_stop)
    stop_session_by_id(sid, reason);
}

void WebRtcGatewayService::set_active_local(bool active, const nlohmann::json& meta) {
  active_.store(active, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = published_state_.find("active");
    if (it == published_state_.end() || it->second != active) {
      published_state_["active"] = active;
      f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "active", active, "cmd", meta);
    }
  }
}

void WebRtcGatewayService::on_activate(const nlohmann::json& meta) {
  set_active_local(true, meta);
}
void WebRtcGatewayService::on_deactivate(const nlohmann::json& meta) {
  set_active_local(false, meta);
}
void WebRtcGatewayService::on_set_active(bool active, const nlohmann::json& meta) {
  set_active_local(active, meta);
}

bool WebRtcGatewayService::on_set_state(const std::string& node_id, const std::string& field,
                                        const nlohmann::json& value, const nlohmann::json& meta,
                                        std::string& error_code, std::string& error_message) {
  if (node_id != cfg_.service_id) {
    error_code = "INVALID_ARGS";
    error_message = "nodeId must equal serviceId for service node state";
    return false;
  }

  const std::string f = field;
  if (f == "active") {
    if (!value.is_boolean()) {
      error_code = "INVALID_VALUE";
      error_message = "active must be boolean";
      return false;
    }
    set_active_local(value.get<bool>(), meta);
    return true;
  }
  if (f == "wsPort") {
    if (!value.is_number_integer() && !value.is_number()) {
      error_code = "INVALID_VALUE";
      error_message = "wsPort must be a number";
      return false;
    }
    const auto port_i = static_cast<int>(value.get<double>());
    if (port_i <= 0 || port_i > 65535) {
      error_code = "INVALID_VALUE";
      error_message = "wsPort must be in 1..65535";
      return false;
    }
    cfg_.ws_port = static_cast<std::uint16_t>(port_i);
    std::string err;
    if (!restart_ws(err)) {
      error_code = "INTERNAL";
      error_message = "restart ws failed: " + err;
      return false;
    }
    json write_value = static_cast<int>(cfg_.ws_port);
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      auto it = published_state_.find("wsPort");
      if (it == published_state_.end() || it->second != write_value) {
        published_state_["wsPort"] = write_value;
        f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "wsPort", write_value, "endpoint", meta);
      }
      const auto url = ws_url(cfg_.ws_port);
      auto it2 = published_state_.find("wsUrl");
      if (it2 == published_state_.end() || it2->second != url) {
        published_state_["wsUrl"] = url;
        f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "wsUrl", url, "endpoint", meta);
      }
    }
    return true;
  }

  error_code = "UNKNOWN_FIELD";
  error_message = "unknown state field";
  return false;
}

bool WebRtcGatewayService::on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta,
                                           std::string& error_code, std::string& error_message) {
  error_code.clear();
  error_message.clear();

  try {
    if (!graph_obj.is_object() || !graph_obj.contains("nodes") || !graph_obj["nodes"].is_array()) {
      return true;
    }

    json service_node;
    for (const auto& n : graph_obj["nodes"]) {
      if (!n.is_object())
        continue;
      const std::string nid = n.value("nodeId", "");
      if (nid != cfg_.service_id)
        continue;
      bool is_service_snapshot = true;
      if (n.contains("operatorClass") && !n["operatorClass"].is_null()) {
        try {
          const std::string oc = n["operatorClass"].is_string() ? n["operatorClass"].get<std::string>() : "";
          if (!oc.empty())
            is_service_snapshot = false;
        } catch (...) {}
      }
      if (!is_service_snapshot)
        continue;
      service_node = n;
      break;
    }

    if (!service_node.is_object() || !service_node.contains("stateValues") ||
        !service_node["stateValues"].is_object()) {
      return true;
    }

    json meta2 = meta;
    if (!meta2.is_object())
      meta2 = json::object();
    meta2["via"] = "rungraph";
    meta2["graphId"] = graph_obj.value("graphId", "");

    const auto& values = service_node["stateValues"];
    for (auto it = values.begin(); it != values.end(); ++it) {
      const std::string field = it.key();
      if (field != "active" && field != "wsPort")
        continue;
      std::string ec;
      std::string em;
      (void)on_set_state(cfg_.service_id, field, it.value(), meta2, ec, em);
    }

    publish_static_state();
    publish_dynamic_state();
  } catch (...) {}

  return true;
}

bool WebRtcGatewayService::on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                                      nlohmann::json& result, std::string& error_code, std::string& error_message) {
  (void)meta;
  error_code.clear();
  error_message.clear();

  if (call == "broadcast") {
    if (!args.is_object() || !args.contains("text") || !args["text"].is_string()) {
      error_code = "INVALID_ARGS";
      error_message = "missing text";
      return false;
    }
    ws_.broadcastText(args["text"].get<std::string>());
    result = json{{"ok", true}};
    return true;
  }
  if (call == "send") {
    if (!args.is_object() || !args.contains("clientId") || !args["clientId"].is_string() || !args.contains("text") ||
        !args["text"].is_string()) {
      error_code = "INVALID_ARGS";
      error_message = "missing clientId/text";
      return false;
    }
    const bool ok = ws_.sendText(args["clientId"].get<std::string>(), args["text"].get<std::string>());
    result = json{{"ok", ok}};
    return ok;
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void WebRtcGatewayService::publish_static_state() {
  const json meta = json{{"via", "startup"}};

  std::vector<std::pair<std::string, json>> updates;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto want = [&](const std::string& field, const json& v) {
      auto it = published_state_.find(field);
      if (it != published_state_.end() && it->second == v)
        return;
      published_state_[field] = v;
      updates.emplace_back(field, v);
    };
    want("serviceClass", cfg_.service_class);
    want("active", active_.load());
    want("wsPort", static_cast<int>(cfg_.ws_port));
    want("wsUrl", ws_url(cfg_.ws_port));
  }

  for (const auto& [field, v] : updates) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
  }
}

void WebRtcGatewayService::publish_dynamic_state() {
  const json meta = json{{"via", "periodic"}};
  const auto cnt = static_cast<int>(ws_.connectionCount());
  std::vector<std::pair<std::string, json>> updates;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto want = [&](const std::string& field, const json& v) {
      auto it = published_state_.find(field);
      if (it != published_state_.end() && it->second == v)
        return;
      published_state_[field] = v;
      updates.emplace_back(field, v);
    };
    want("connections", cnt);
  }
  for (const auto& [field, v] : updates) {
    f8::cppsdk::kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, field, v, "runtime", meta);
  }
}

json WebRtcGatewayService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.webrtc.gateway";
  service["label"] = "WebRTC Gateway";
  service["version"] = "0.0.1";
  service["description"] = "Localhost WebRTC signaling gateway (WS) for browser capture streams.";
  service["stateFields"] = json::array({
      state_field("active", schema_boolean(), "rw", "Active", "Accept and forward signaling when true.", true),
      state_field("wsPort", schema_integer(), "rw", "WS Port", "Localhost websocket port.", true),
      state_field("wsUrl", schema_string(), "ro", "WS URL", "Computed ws://127.0.0.1:<port>/ws"),
      state_field("connections", schema_integer(), "ro", "Connections", "Current websocket client count.", true),
  });

  service["dataOutPorts"] = json::array({
      json{{"name", "signalRx"},
           {"valueSchema", schema_object(json{{"clientId", schema_string()},
                                              {"text", schema_string()},
                                              {"wsUrl", schema_string()},
                                              {"ts", schema_integer()},
                                              {"json", json{{"type", "any"}}}},
                                         json::array({"clientId", "text", "ts"}))},
           {"description", "Inbound websocket signaling messages from browser."},
           {"required", false}},
  });

  service["commands"] = json::array({
      json{{"name", "broadcast"},
           {"description", "Broadcast a text frame to all websocket clients."},
           {"params", json::array({json{{"name", "text"}, {"valueSchema", schema_string()}, {"required", true}}})}},
      json{{"name", "send"},
           {"description", "Send a text frame to one websocket client."},
           {"params", json::array({json{{"name", "clientId"}, {"valueSchema", schema_string()}, {"required", true}},
                                   json{{"name", "text"}, {"valueSchema", schema_string()}, {"required", true}}})}},
  });

  json out;
  out["service"] = service;
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::webrtc_gateway
