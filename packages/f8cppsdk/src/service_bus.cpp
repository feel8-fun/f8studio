#include "f8cppsdk/service_bus.h"

#include <chrono>
#include <cstring>
#include <utility>

#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/rungraph_routes.h"
#include "f8cppsdk/state_kv.h"

namespace f8::cppsdk {

using json = nlohmann::json;

ServiceBus::ServiceBus(Config cfg) : cfg_(std::move(cfg)) {}

ServiceBus::~ServiceBus() {
  stop();
}

void ServiceBus::add_lifecycle_node(LifecycleNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(lifecycle_mu_);
  lifecycle_nodes_.push_back(node);
}

void ServiceBus::add_stateful_node(StatefulNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  stateful_nodes_.push_back(node);
}

void ServiceBus::add_data_node(DataReceivableNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  data_nodes_.push_back(node);
}

void ServiceBus::add_set_state_node(SetStateHandlerNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  set_state_nodes_.push_back(node);
}

void ServiceBus::add_rungraph_node(RungraphHandlerNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  rungraph_nodes_.push_back(node);
}

void ServiceBus::add_command_node(CommandableNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  command_nodes_.push_back(node);
}

std::size_t ServiceBus::drain_main_thread(std::size_t max_tasks) {
  return main_thread_.drain(max_tasks);
}

void ServiceBus::apply_data_routes_from_rungraph(const json& graph_obj) {
  auto new_routes = parse_cross_service_data_routes(graph_obj, cfg_.service_id);

  std::lock_guard<std::mutex> lock(data_mu_);

  // Unsubscribe removed subjects.
  for (auto it = data_subs_.begin(); it != data_subs_.end();) {
    if (new_routes.find(it->first) != new_routes.end()) {
      ++it;
      continue;
    }
    try {
      it->second.unsubscribe();
    } catch (...) {
    }
    it = data_subs_.erase(it);
  }

  // Subscribe new subjects.
  for (const auto& kv : new_routes) {
    const std::string& subject = kv.first;
    if (data_subs_.find(subject) != data_subs_.end()) {
      continue;
    }
    auto sub = nats_.subscribe(subject, [this, subject](natsMsg* msg) {
      if (msg == nullptr) return;
      const void* data = natsMsg_GetData(msg);
      const int len = natsMsg_GetDataLength(msg);
      if (data == nullptr || len < 0) return;

      json payload = json::object();
      try {
        const std::string s(reinterpret_cast<const char*>(data), static_cast<std::size_t>(len));
        payload = json::parse(s, nullptr, false);
      } catch (...) {
        return;
      }
      if (!payload.is_object()) return;

      const json value = payload.contains("value") ? payload["value"] : json();
      std::int64_t ts_ms = 0;
      try {
        if (payload.contains("ts")) ts_ms = payload["ts"].get<std::int64_t>();
      } catch (...) {
        ts_ms = 0;
      }

      json meta = payload;
      try {
        if (meta.is_object()) meta.erase("value");
      } catch (...) {
        meta = json::object();
      }

      // Enrich meta for consumers.
      try {
        meta["subject"] = subject;
      } catch (...) {
      }

      main_thread_.post([this, subject, value, ts_ms, meta]() {
        std::vector<DataRoute> routes;
        {
          std::lock_guard<std::mutex> lock(data_mu_);
          auto it = data_routes_by_subject_.find(subject);
          if (it != data_routes_by_subject_.end()) routes = it->second;
        }

        if (routes.empty()) return;

        std::vector<DataReceivableNode*> nodes;
        {
          std::lock_guard<std::mutex> lock(handlers_mu_);
          nodes = data_nodes_;
        }
        if (nodes.empty()) return;

        for (const auto& r : routes) {
          json m = meta;
          try {
            m["fromServiceId"] = r.from_service_id;
            m["fromNodeId"] = r.from_node_id;
            m["fromPort"] = r.from_port;
          } catch (...) {
          }
          for (auto* n : nodes) {
            if (!n) continue;
            try {
              n->on_data(r.to_node_id, r.to_port, value, ts_ms, m);
            } catch (...) {
              continue;
            }
          }
        }
      });
    });
    if (sub.valid()) {
      data_subs_.emplace(subject, std::move(sub));
    }
  }

  data_routes_by_subject_ = std::move(new_routes);
}

bool ServiceBus::start() {
  stop();

  cfg_.service_id = ensure_token(cfg_.service_id, "service_id");
  if (!nats_.connect(cfg_.nats_url)) {
    return false;
  }

  KvConfig kvc;
  kvc.bucket = kv_bucket_for_service(cfg_.service_id);
  kvc.memory_storage = cfg_.kv_memory_storage;
  if (!kv_.open_or_create(nats_.jetstream(), kvc)) {
    nats_.close();
    return false;
  }

  ctrl_ = std::make_unique<ServiceControlPlaneServer>(
      ServiceControlPlaneServer::Config{cfg_.service_id, cfg_.nats_url}, &nats_, &kv_, this);
  if (!ctrl_->start()) {
    ctrl_.reset();
    kv_.close();
    nats_.close();
    return false;
  }

  load_active_from_kv();

  // Watch node state changes and forward to stateful nodes (best-effort).
  // Key format: nodes.<node_id>.state.<field>
  (void)kv_.watch(
      "nodes.>",
      [this](const std::string& key, const std::vector<std::uint8_t>& bytes) {
        constexpr const char* kPrefix = "nodes.";
        constexpr const char* kStateMarker = ".state.";

        if (key.rfind(kPrefix, 0) != 0) return;
        const std::size_t marker = key.find(kStateMarker);
        if (marker == std::string::npos) return;

        const std::size_t node_begin = std::strlen(kPrefix);
        const std::size_t node_end = marker;
        if (node_end <= node_begin) return;
        const std::string node_id = key.substr(node_begin, node_end - node_begin);

        const std::size_t field_begin = marker + std::strlen(kStateMarker);
        if (field_begin >= key.size()) return;
        const std::string field = key.substr(field_begin);

        nlohmann::json payload = nlohmann::json::object();
        try {
          const std::string s(reinterpret_cast<const char*>(bytes.data()), bytes.size());
          payload = nlohmann::json::parse(s, nullptr, false);
        } catch (...) {
          return;
        }
        if (!payload.is_object()) return;

        // Avoid loopback: ignore updates authored by this service process.
        // External state changes (other actors) still flow through.
        try {
          const std::string actor = payload.value("actor", std::string());
          if (!actor.empty() && actor == cfg_.service_id) {
            return;
          }
        } catch (...) {
        }

        const nlohmann::json value = payload.contains("value") ? payload["value"] : nlohmann::json();
        std::int64_t ts_ms = 0;
        try {
          if (payload.contains("ts")) ts_ms = payload["ts"].get<std::int64_t>();
        } catch (...) {
          ts_ms = 0;
        }
        nlohmann::json meta = payload;
        try {
          if (meta.is_object()) meta.erase("value");
        } catch (...) {
          meta = nlohmann::json::object();
        }

        main_thread_.post([this, node_id, field, value, ts_ms, meta]() {
          std::vector<StatefulNode*> nodes;
          {
            std::lock_guard<std::mutex> lock(handlers_mu_);
            nodes = stateful_nodes_;
          }
          for (auto* n : nodes) {
            if (!n) continue;
            try {
              n->on_state(node_id, field, value, ts_ms, meta);
            } catch (...) {
              continue;
            }
          }
        });
      },
      true);

  // Seed identity fields (best-effort). Specs should declare these as readonly.
  try {
    (void)kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "svcId", cfg_.service_id, "runtime",
                            json{{"builtin", true}});
  } catch (...) {}

  // Announce readiness after endpoints are up.
  (void)kv_set_ready(kv_, true);
  spdlog::info("service_bus started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void ServiceBus::stop() {
  (void)kv_set_ready(kv_, false);

  if (ctrl_) {
    try {
      ctrl_->stop();
    } catch (...) {}
  }
  ctrl_.reset();

  try {
    kv_.stop_watch();
  } catch (...) {}
  {
    std::lock_guard<std::mutex> lock(data_mu_);
    for (auto& kv : data_subs_) {
      try {
        kv.second.unsubscribe();
      } catch (...) {
      }
    }
    data_subs_.clear();
    data_routes_by_subject_.clear();
  }
  try {
    main_thread_.clear();
  } catch (...) {}
  kv_.close();
  nats_.close();
}

void ServiceBus::wait_terminate() {
  std::unique_lock<std::mutex> lock(term_mu_);
  term_cv_.wait(lock, [this]() { return terminate_.load(std::memory_order_acquire); });
}

void ServiceBus::set_active_local(bool active, const json& meta, const std::string& source) {
  active_.store(active, std::memory_order_release);

  // Persist `nodes.<serviceId>.state.active` (mirror pysdk).
  const json extra = meta.is_object() ? meta : json::object();
  (void)kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "active", active, source, extra);

  std::vector<LifecycleNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    nodes = lifecycle_nodes_;
  }
  for (const auto& n : nodes) {
    try {
      if (n) n->on_lifecycle(active, meta);
    } catch (...) {
      continue;
    }
  }
}

bool ServiceBus::is_active() const {
  return active_.load(std::memory_order_acquire);
}

void ServiceBus::on_activate(const json& meta) {
  set_active_local(true, meta, "cmd");
}
void ServiceBus::on_deactivate(const json& meta) {
  set_active_local(false, meta, "cmd");
}
void ServiceBus::on_set_active(bool active, const json& meta) {
  set_active_local(active, meta, "cmd");
}

bool ServiceBus::on_set_state(const std::string& node_id, const std::string& field, const json& value, const json& meta,
                              std::string& error_code, std::string& error_message) {
  if (field == "active") {
    if (!value.is_boolean()) {
      error_code = "INVALID_ARGS";
      error_message = "active must be boolean";
      return false;
    }
    set_active_local(value.get<bool>(), meta, "endpoint");
    return true;
  }

  std::vector<SetStateHandlerNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = set_state_nodes_;
  }
  for (auto* n : nodes) {
    if (!n) continue;
    error_code.clear();
    error_message.clear();
    try {
      if (n->on_set_state(node_id, field, value, meta, error_code, error_message)) {
        return true;
      }
    } catch (...) {
      error_code = "INTERNAL_ERROR";
      error_message = "on_set_state threw";
      return false;
    }
  }
  error_code = "NOT_SUPPORTED";
  error_message = "set_state not supported";
  return false;
}

bool ServiceBus::on_set_rungraph(const json& graph_obj, const json& meta, std::string& error_code,
                                 std::string& error_message) {
  try {
    apply_data_routes_from_rungraph(graph_obj);
  } catch (...) {
  }
  std::vector<RungraphHandlerNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = rungraph_nodes_;
  }
  if (nodes.empty()) {
    // Allow deploy even if unhandled (runtime-only services).
    return true;
  }
  for (auto* n : nodes) {
    if (!n) continue;
    error_code.clear();
    error_message.clear();
    try {
      if (n->on_set_rungraph(graph_obj, meta, error_code, error_message)) {
        return true;
      }
    } catch (...) {
      error_code = "INTERNAL_ERROR";
      error_message = "on_set_rungraph threw";
      return false;
    }
  }
  if (error_code.empty()) error_code = "NOT_SUPPORTED";
  if (error_message.empty()) error_message = "set_rungraph not supported";
  return false;
}

bool ServiceBus::on_command(const std::string& call, const json& args, const json& meta, json& result,
                            std::string& error_code, std::string& error_message) {
  if (call == "terminate" || call == "quit") {
    spdlog::info("service_bus terminate requested serviceId={}", cfg_.service_id);
    terminate_.store(true, std::memory_order_release);
    term_cv_.notify_all();
    result = json::object();
    result["terminating"] = true;
    return true;
  }

  std::vector<CommandableNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = command_nodes_;
  }
  for (auto* n : nodes) {
    if (!n) continue;
    error_code.clear();
    error_message.clear();
    try {
      if (n->on_command(call, args, meta, result, error_code, error_message)) {
        return true;
      }
    } catch (...) {
      error_code = "INTERNAL_ERROR";
      error_message = "on_command threw";
      return false;
    }
  }
  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void ServiceBus::load_active_from_kv() {
  try {
    const auto key = kv_key_node_state(cfg_.service_id, "active");
    const auto raw = kv_.get(key);
    if (!raw.has_value()) {
      return;
    }
    const std::string s(reinterpret_cast<const char*>(raw->data()), raw->size());
    json payload = json::parse(s, nullptr, false);
    if (!payload.is_object() || !payload.contains("value")) {
      return;
    }
    const json v = payload["value"];
    if (!v.is_boolean()) {
      return;
    }
    set_active_local(v.get<bool>(), json::object({{"init", true}}), "kv");
  } catch (...) {
    return;
  }
}

}  // namespace f8::cppsdk
