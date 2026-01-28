#include "f8cppsdk/service_control_plane_server.h"

#include <nats/nats.h>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

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

ServiceControlPlaneServer::~ServiceControlPlaneServer() {
  stop();
}

bool ServiceControlPlaneServer::start() {
  if (client_ == nullptr || kv_ == nullptr || handler_ == nullptr) {
    return false;
  }
  const auto sid = ensure_token(cfg_.service_id, "service_id");

  if (client_->raw() == nullptr) {
    spdlog::error("micro endpoints require an active NATS connection");
    return false;
  }

  microServiceConfig sc{};
  const auto micro_name = svc_micro_name(sid);
  sc.Name = micro_name.c_str();
  sc.Version = "0.0.1";
  sc.Description = "F8 service runtime control plane (lifecycle + cmd + state + rungraph).";
  const char* md_kv[] = {"serviceId", sid.c_str()};
  sc.Metadata.List = md_kv;
  sc.Metadata.Count = 1;
  sc.State = this;

  microError* err = micro_AddService(&micro_, client_->raw(), &sc);
  if (err != nullptr) {
    char buf[256] = {};
    spdlog::error("micro_AddService failed: {}", microError_String(err, buf, sizeof(buf)));
    microError_Destroy(err);
    micro_ = nullptr;
    return false;
  }

  auto add_ep = [&](const char* name, const std::string& subject) -> bool {
    microEndpointConfig ec{};
    ec.Name = name;
    ec.Subject = subject.c_str();
    ec.Handler = &ServiceControlPlaneServer::on_micro_request;
    ec.State = const_cast<char*>(name);  // stable, points to string literal
    microError* e = microService_AddEndpoint(micro_, &ec);
    if (e != nullptr) {
      char buf[256] = {};
      spdlog::error("microService_AddEndpoint failed ep={} subject={} err={}", name, subject,
                    microError_String(e, buf, sizeof(buf)));
      microError_Destroy(e);
      return false;
    }
    return true;
  };

  const bool ok = add_ep("activate", svc_endpoint_subject(sid, "activate")) &&
                  add_ep("deactivate", svc_endpoint_subject(sid, "deactivate")) &&
                  add_ep("set_active", svc_endpoint_subject(sid, "set_active")) &&
                  add_ep("status", svc_endpoint_subject(sid, "status")) &&
                  add_ep("terminate", svc_endpoint_subject(sid, "terminate")) &&
                  add_ep("quit", svc_endpoint_subject(sid, "quit")) && add_ep("cmd", cmd_channel_subject(sid)) &&
                  add_ep("set_state", svc_endpoint_subject(sid, "set_state")) &&
                  add_ep("set_rungraph", svc_endpoint_subject(sid, "set_rungraph"));
  if (!ok) {
    stop();
    return false;
  }
  return true;
}

void ServiceControlPlaneServer::stop() {
  if (micro_ != nullptr) {
    microError* err = microService_Destroy(micro_);
    if (err != nullptr) {
      char buf[256] = {};
      spdlog::warn("microService_Destroy failed: {}", microError_String(err, buf, sizeof(buf)));
      microError_Destroy(err);
    }
    micro_ = nullptr;
  }
}

microError* ServiceControlPlaneServer::on_micro_request(microRequest* req) {
  if (req == nullptr) {
    return nullptr;
  }
  auto* self = static_cast<ServiceControlPlaneServer*>(microRequest_GetServiceState(req));
  const char* endpoint = static_cast<const char*>(microRequest_GetEndpointState(req));
  if (self == nullptr || endpoint == nullptr) {
    return nullptr;
  }
  self->handle_request(req, std::string(endpoint));
  return nullptr;
}

void ServiceControlPlaneServer::respond(microRequest* req, const std::string& req_id, bool ok, const json& result,
                                        const std::string& err_code, const std::string& err_message) {
  if (req == nullptr) {
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
  (void)microRequest_Respond(req, out.data(), out.size());
}

void ServiceControlPlaneServer::handle_request(microRequest* msg, const std::string& endpoint) {
  if (client_ == nullptr || handler_ == nullptr || kv_ == nullptr || msg == nullptr) {
    return;
  }
  const void* data = microRequest_GetData(msg);
  const int len = microRequest_GetDataLength(msg);
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
      result = json{{"serviceId", cfg_.service_id}, {"active", handler_->is_active()}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "terminate" || endpoint == "quit") {
      spdlog::info("{} requested serviceId={}", endpoint, cfg_.service_id);
      json out;
      bool ok = handler_->on_command("terminate", env.args, env.meta, out, err_code, err_msg);
      if (!ok) {
        respond(msg, env.req_id, false, json(nullptr), err_code, err_msg);
        return;
      }
      respond(msg, env.req_id, true, json{{"terminating", true}}, "", "");
      return;
    }
    if (endpoint == "set_state") {
      std::string node_id_s;
      std::string field_s;
      if (env.args.contains("nodeId") && env.args["nodeId"].is_string())
        node_id_s = env.args["nodeId"].get<std::string>();
      if (node_id_s.empty() && env.raw.contains("nodeId") && env.raw["nodeId"].is_string())
        node_id_s = env.raw["nodeId"].get<std::string>();
      if (env.args.contains("field") && env.args["field"].is_string())
        field_s = env.args["field"].get<std::string>();
      if (field_s.empty() && env.raw.contains("field") && env.raw["field"].is_string())
        field_s = env.raw["field"].get<std::string>();
      json value;
      if (env.args.contains("value"))
        value = env.args["value"];
      else if (env.raw.contains("value"))
        value = env.raw["value"];
      else
        value = json(nullptr);
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
      if (env.args.contains("graph") && env.args["graph"].is_object())
        graph_obj = env.args["graph"];
      else if (env.raw.contains("graph") && env.raw["graph"].is_object())
        graph_obj = env.raw["graph"];
      else if (env.raw.is_object() && env.raw.contains("nodes") && env.raw.contains("edges"))
        graph_obj = env.raw;
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
      if (!persisted.contains("meta") || !persisted["meta"].is_object())
        persisted["meta"] = json::object();
      persisted["meta"]["ts"] = now_ms();
      const auto bytes = persisted.dump();
      kv_->put(kv_key_rungraph(), bytes.data(), bytes.size());
      result = json{{"graphId", persisted.value("graphId", "")}};
      respond(msg, env.req_id, true, result, "", "");
      return;
    }
    if (endpoint == "cmd") {
      std::string call;
      if (env.raw.contains("call") && env.raw["call"].is_string())
        call = env.raw["call"].get<std::string>();
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
