#include "f8cppsdk/service_control_plane_server.h"

#include <nats/nats.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/service_control_plane.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppsdk {

using json = nlohmann::json;

namespace {

std::string new_req_id() {
  // cnats provides nuid, but keep simple.
  return std::to_string(static_cast<long long>(now_ms()));
}

struct Envelope {
  std::string req_id;
  json raw;
  json args;
  json meta;
};

Envelope parse_envelope(const void* data, std::size_t len) {
  Envelope out;
  out.raw = json::object();
  if (data != nullptr && len > 0) {
    try {
      out.raw = json::parse(std::string(static_cast<const char*>(data), len), nullptr, false);
    } catch (...) {
      out.raw = json::object();
    }
    if (!out.raw.is_object()) {
      out.raw = json::object();
    }
  }
  if (out.raw.contains("reqId") && out.raw["reqId"].is_string()) {
    out.req_id = out.raw["reqId"].get<std::string>();
  }
  if (out.req_id.empty()) {
    out.req_id = new_req_id();
  }
  if (out.raw.contains("args") && out.raw["args"].is_object()) {
    out.args = out.raw["args"];
  } else {
    out.args = json::object();
  }
  if (out.raw.contains("meta") && out.raw["meta"].is_object()) {
    out.meta = out.raw["meta"];
  } else {
    out.meta = json::object();
  }
  return out;
}

}  // namespace

ServiceControlPlaneServer::ServiceControlPlaneServer(Config cfg, NatsClient* client, KvStore* kv,
                                                     ServiceControlHandler* handler)
    : cfg_(std::move(cfg)), client_(client), kv_(kv), handler_(handler) {}

ServiceControlPlaneServer::~ServiceControlPlaneServer() { stop(); }

bool ServiceControlPlaneServer::start() {
  if (client_ == nullptr || kv_ == nullptr || handler_ == nullptr) {
    return false;
  }
  const auto sid = ensure_token(cfg_.service_id, "service_id");

  sub_activate_ = client_->subscribe(svc_endpoint_subject(sid, "activate"), [this](natsMsg* msg) {
    handle_request(msg, "activate");
  });
  sub_deactivate_ = client_->subscribe(svc_endpoint_subject(sid, "deactivate"), [this](natsMsg* msg) {
    handle_request(msg, "deactivate");
  });
  sub_set_active_ = client_->subscribe(svc_endpoint_subject(sid, "set_active"), [this](natsMsg* msg) {
    handle_request(msg, "set_active");
  });
  sub_status_ = client_->subscribe(svc_endpoint_subject(sid, "status"), [this](natsMsg* msg) { handle_request(msg, "status"); });
  sub_set_state_ = client_->subscribe(svc_endpoint_subject(sid, "set_state"), [this](natsMsg* msg) {
    handle_request(msg, "set_state");
  });
  sub_set_rungraph_ = client_->subscribe(svc_endpoint_subject(sid, "set_rungraph"), [this](natsMsg* msg) {
    handle_request(msg, "set_rungraph");
  });
  sub_cmd_ = client_->subscribe(cmd_channel_subject(sid), [this](natsMsg* msg) { handle_request(msg, "cmd"); });

  return sub_activate_.valid() && sub_deactivate_.valid() && sub_set_active_.valid() && sub_status_.valid() &&
         sub_set_state_.valid() && sub_set_rungraph_.valid() && sub_cmd_.valid();
}

void ServiceControlPlaneServer::stop() {
  sub_activate_.unsubscribe();
  sub_deactivate_.unsubscribe();
  sub_set_active_.unsubscribe();
  sub_status_.unsubscribe();
  sub_set_state_.unsubscribe();
  sub_set_rungraph_.unsubscribe();
  sub_cmd_.unsubscribe();
}

void ServiceControlPlaneServer::respond(natsMsg* req, const std::string& req_id, bool ok, const json& result,
                                       const std::string& err_code, const std::string& err_message) {
  if (client_ == nullptr || req == nullptr) {
    return;
  }
  const char* reply = natsMsg_GetReply(req);
  if (reply == nullptr || std::string(reply).empty()) {
    return;
  }

  json payload;
  payload["reqId"] = req_id;
  payload["ok"] = ok;
  payload["result"] = ok ? result : json(nullptr);
  if (!ok) {
    payload["error"] = json{{"code", err_code.empty() ? "INTERNAL" : err_code}, {"message", err_message}};
  } else {
    payload["error"] = json(nullptr);
  }
  const auto out = payload.dump();
  client_->publish(reply, out.data(), out.size());
}

void ServiceControlPlaneServer::handle_request(natsMsg* msg, const std::string& endpoint) {
  if (client_ == nullptr || handler_ == nullptr || kv_ == nullptr || msg == nullptr) {
    return;
  }
  const void* data = natsMsg_GetData(msg);
  const int len = natsMsg_GetDataLength(msg);
  const auto env = parse_envelope(data, len < 0 ? 0 : static_cast<std::size_t>(len));

  std::string err_code;
  std::string err_msg;
  json result = json::object();

  try {
    if (endpoint == "activate") {
      handler_->on_activate(env.meta);
      result = json{{"active", true}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "deactivate") {
      handler_->on_deactivate(env.meta);
      result = json{{"active", false}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "set_active") {
      bool active = false;
      if (env.args.contains("active")) {
        active = env.args["active"].get<bool>();
      } else if (env.raw.contains("active")) {
        active = env.raw["active"].get<bool>();
      } else {
        respond(msg, env.req_id, false, json(nullptr), "INVALID_ARGS", "missing active");
        return;
      }
      handler_->on_set_active(active, env.meta);
      result = json{{"active", active}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "status") {
      result = json{{"serviceId", cfg_.service_id}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "set_state") {
      std::string node_id_s;
      std::string field_s;
      if (env.args.contains("nodeId") && env.args["nodeId"].is_string()) node_id_s = env.args["nodeId"].get<std::string>();
      if (node_id_s.empty() && env.raw.contains("nodeId") && env.raw["nodeId"].is_string()) node_id_s = env.raw["nodeId"].get<std::string>();
      if (env.args.contains("field") && env.args["field"].is_string()) field_s = env.args["field"].get<std::string>();
      if (field_s.empty() && env.raw.contains("field") && env.raw["field"].is_string()) field_s = env.raw["field"].get<std::string>();
      json value;
      if (env.args.contains("value")) value = env.args["value"];
      else if (env.raw.contains("value")) value = env.raw["value"];
      else value = json(nullptr);
      if (node_id_s.empty() || field_s.empty()) {
        respond(msg, env.req_id, false, json(nullptr), "INVALID_ARGS", "missing nodeId/field");
        return;
      }
      bool ok = handler_->on_set_state(node_id_s, field_s, value, env.meta, err_code, err_msg);
      if (!ok) {
        respond(msg, env.req_id, false, json(nullptr), err_code, err_msg);
        return;
      }
      result = json{{"nodeId", node_id_s}, {"field", field_s}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "set_rungraph") {
      json graph_obj;
      if (env.args.contains("graph") && env.args["graph"].is_object()) graph_obj = env.args["graph"];
      else if (env.raw.contains("graph") && env.raw["graph"].is_object()) graph_obj = env.raw["graph"];
      else if (env.raw.is_object() && env.raw.contains("nodes") && env.raw.contains("edges")) graph_obj = env.raw;
      else {
        respond(msg, env.req_id, false, json(nullptr), "INVALID_ARGS", "missing graph");
        return;
      }
      bool ok = handler_->on_set_rungraph(graph_obj, env.meta, err_code, err_msg);
      if (!ok) {
        respond(msg, env.req_id, false, json(nullptr), err_code, err_msg);
        return;
      }
      // Persist to KV (mirror python behavior of adding meta.ts).
      json persisted = graph_obj;
      if (!persisted.contains("meta") || !persisted["meta"].is_object()) persisted["meta"] = json::object();
      persisted["meta"]["ts"] = now_ms();
      const auto bytes = persisted.dump();
      kv_->put(kv_key_rungraph(), bytes.data(), bytes.size());
      result = json{{"graphId", persisted.value("graphId", "")}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "cmd") {
      std::string call;
      if (env.raw.contains("call") && env.raw["call"].is_string()) call = env.raw["call"].get<std::string>();
      if (call.empty()) {
        respond(msg, env.req_id, false, json(nullptr), "INVALID_ARGS", "missing call");
        return;
      }
      json out;
      bool ok = handler_->on_command(call, env.args, env.meta, out, err_code, err_msg);
      if (!ok) {
        respond(msg, env.req_id, false, json(nullptr), err_code, err_msg);
        return;
      }
      respond(msg, env.req_id, true, out, "", "");
      return;
    }
  } catch (const std::exception& ex) {
    respond(msg, env.req_id, false, json(nullptr), "INTERNAL", ex.what());
    return;
  } catch (...) {
    respond(msg, env.req_id, false, json(nullptr), "INTERNAL", "unknown error");
    return;
  }

  respond(msg, env.req_id, false, json(nullptr), "NOT_FOUND", "unknown endpoint");
}

}  // namespace f8::cppsdk
