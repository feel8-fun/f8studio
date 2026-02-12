#include "f8cppsdk/service_bus.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <spdlog/spdlog.h>

#include "f8cppsdk/data_bus.h"
#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/generated/protocol_models.h"
#include "f8cppsdk/rungraph_routes.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppsdk {

using json = nlohmann::json;

namespace {

bool state_debug_enabled() {
  const char* v = std::getenv("F8_STATE_DEBUG");
  if (v == nullptr) return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

std::int64_t coerce_inbound_ts_ms(const json& payload, std::int64_t default_ts_ms) {
  auto read_int = [&](const char* key) -> std::optional<std::int64_t> {
    try {
      if (!payload.is_object() || !payload.contains(key)) return std::nullopt;
      const auto& v = payload.at(key);
      if (v.is_number_integer()) return v.get<std::int64_t>();
      if (v.is_number_float()) return static_cast<std::int64_t>(v.get<double>());
      if (v.is_string()) return std::stoll(v.get<std::string>());
    } catch (...) {
      return std::nullopt;
    }
    return std::nullopt;
  };

  std::optional<std::int64_t> ts = read_int("ts");
  if (!ts.has_value()) ts = read_int("ts_ms");
  if (!ts.has_value()) ts = read_int("tsMs");

  std::int64_t t = ts.value_or(default_ts_ms);
  if (t <= 0) return default_ts_ms;

  // Heuristics matching pysdk:
  // - seconds ~1e9, ms ~1e12 (2026), micros ~1e15, nanos ~1e18
  if (t < 100'000'000'000LL) return t * 1000LL;
  if (t >= 100'000'000'000'000'000LL) return t / 1'000'000LL;
  if (t >= 100'000'000'000'000LL) return t / 1000LL;
  return t;
}

bool state_origin_allows_access(const std::string& origin, const std::string& access) {
  // origin: "external" | "runtime" | "rungraph" | "system"
  // access: "rw" | "ro" | "wo"
  if (origin == "system") return true;
  if (origin == "runtime") return (access == "rw" || access == "ro");
  if (origin == "rungraph") return (access == "rw" || access == "wo");
  if (origin == "external") return (access == "rw" || access == "wo");
  return false;
}

std::string access_to_string(f8::cppsdk::generated::F8StateAccess a) {
  switch (a) {
    case f8::cppsdk::generated::F8StateAccess::rw:
      return "rw";
    case f8::cppsdk::generated::F8StateAccess::ro:
      return "ro";
    case f8::cppsdk::generated::F8StateAccess::wo:
      return "wo";
  }
  return "";
}

struct StateEdgeKey {
  std::string service_id;
  std::string node_id;
  std::string field;
  bool operator==(const StateEdgeKey& other) const {
    return service_id == other.service_id && node_id == other.node_id && field == other.field;
  }
};
struct StateEdgeKeyHash {
  std::size_t operator()(const StateEdgeKey& k) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(k.service_id);
    std::size_t h2 = std::hash<std::string>{}(k.node_id);
    std::size_t h3 = std::hash<std::string>{}(k.field);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

void validate_state_edges_or_throw(const f8::cppsdk::generated::F8RuntimeGraph& graph) {
  using namespace f8::cppsdk::generated;

  const auto edges = graph.edges.value_or(std::vector<F8Edge>{});
  std::unordered_map<StateEdgeKey, std::vector<StateEdgeKey>, StateEdgeKeyHash> out;
  std::unordered_map<StateEdgeKey, int, StateEdgeKeyHash> inbound_count;
  std::unordered_map<StateEdgeKey, StateEdgeKey, StateEdgeKeyHash> upstream_by_target;
  std::vector<StateEdgeKey> nodes;

  auto add_node = [&](const StateEdgeKey& k) {
    nodes.push_back(k);
  };

  for (const auto& e : edges) {
    if (e.kind != F8EdgeKindEnum::state) continue;
    const std::string from_sid = e.fromServiceId;
    const std::string to_sid = e.toServiceId;
    const std::string from_op = e.fromOperatorId.value_or("");
    const std::string to_op = e.toOperatorId.value_or("");
    const std::string from_field = e.fromPort;
    const std::string to_field = e.toPort;
    if (from_sid.empty() || to_sid.empty() || from_op.empty() || to_op.empty() || from_field.empty() || to_field.empty()) {
      continue;
    }

    StateEdgeKey from{from_sid, from_op, from_field};
    StateEdgeKey to{to_sid, to_op, to_field};

    auto it_prev = upstream_by_target.find(to);
    if (it_prev != upstream_by_target.end()) {
      const auto& prev = it_prev->second;
      if (!(prev == from)) {
        throw std::runtime_error("multiple upstreams for state field: " + to_sid + "." + to_op + "." + to_field);
      }
    } else {
      upstream_by_target.emplace(to, from);
    }

    out[from].push_back(to);
    inbound_count[to] = inbound_count[to] + 1;
    add_node(from);
    add_node(to);
  }

  if (out.empty()) return;

  std::unordered_map<StateEdgeKey, bool, StateEdgeKeyHash> visiting;
  std::unordered_map<StateEdgeKey, bool, StateEdgeKeyHash> visited;
  std::unordered_map<StateEdgeKey, std::optional<StateEdgeKey>, StateEdgeKeyHash> parent;

  auto fmt = [](const StateEdgeKey& k) { return k.service_id + "." + k.node_id + "." + k.field; };

  std::function<std::optional<std::vector<StateEdgeKey>>(const StateEdgeKey&)> dfs;

  dfs = [&](const StateEdgeKey& n) -> std::optional<std::vector<StateEdgeKey>> {
    visiting[n] = true;
    for (const auto& m : out[n]) {
      if (visited[m]) continue;
      if (visiting[m]) {
        std::vector<StateEdgeKey> cyc;
        cyc.push_back(m);
        cyc.push_back(n);
        auto cur = parent[n];
        while (cur.has_value() && !(cur.value() == m)) {
          cyc.push_back(cur.value());
          cur = parent[cur.value()];
        }
        cyc.push_back(m);
        std::reverse(cyc.begin(), cyc.end());
        return cyc;
      }
      parent[m] = n;
      auto r = dfs(m);
      if (r.has_value()) return r;
    }
    visiting[n] = false;
    visited[n] = true;
    return std::nullopt;
  };

  // Roots first.
  std::vector<StateEdgeKey> start;
  for (const auto& n : nodes) {
    if (inbound_count.find(n) == inbound_count.end()) {
      start.push_back(n);
    }
  }
  for (const auto& n : nodes) {
    if (std::find_if(start.begin(), start.end(), [&](const StateEdgeKey& x) { return x == n; }) == start.end()) {
      start.push_back(n);
    }
  }

  for (const auto& n : start) {
    if (visited[n]) continue;
    parent[n] = std::nullopt;
    auto cyc = dfs(n);
    if (cyc.has_value()) {
      std::string msg = "cyclic state-edge loop detected: ";
      for (std::size_t i = 0; i < cyc->size(); ++i) {
        if (i) msg += " -> ";
        msg += fmt(cyc->at(i));
      }
      throw std::runtime_error(msg);
    }
  }
}

}  // namespace

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

  // Build new routing snapshot + input buffers.
  auto next_snapshot = std::make_shared<_DataRoutingSnapshot>();
  auto next_inputs = std::unordered_map<_NodePortKey, std::shared_ptr<_InputBuffer>, _NodePortKeyHash>();

  for (const auto& kv : new_routes) {
    const std::string& subject = kv.first;
    auto& vec = next_snapshot->by_subject[subject];
    vec.reserve(kv.second.size());
    for (const auto& r : kv.second) {
      _NodePortKey key{r.to_node_id, r.to_port};
      auto it = next_inputs.find(key);
      if (it == next_inputs.end()) {
        it = next_inputs.emplace(key, std::make_shared<_InputBuffer>()).first;
      }
      auto& buf = *it->second;
      if (r.strategy == EdgeStrategy::kQueue) {
        buf.strategy = EdgeStrategy::kQueue;
      }
      if (r.timeout_ms > 0) {
        if (buf.timeout_ms <= 0) {
          buf.timeout_ms = r.timeout_ms;
        } else {
          buf.timeout_ms = std::min<std::int64_t>(buf.timeout_ms, r.timeout_ms);
        }
      }

      _RouteRuntime rr;
      rr.to_node_id = r.to_node_id;
      rr.to_port = r.to_port;
      rr.from_service_id = r.from_service_id;
      rr.from_node_id = r.from_node_id;
      rr.from_port = r.from_port;
      rr.strategy = r.strategy;
      rr.timeout_ms = r.timeout_ms;
      rr.buf = it->second;
      vec.push_back(std::move(rr));
    }
  }

  // Unsubscribe removed subjects.
  for (auto it = data_subs_.begin(); it != data_subs_.end();) {
    if (next_snapshot->by_subject.find(it->first) != next_snapshot->by_subject.end()) {
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
  for (const auto& kv : next_snapshot->by_subject) {
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

      json value = payload.contains("value") ? payload["value"] : json();
      const std::int64_t ts_ms = coerce_inbound_ts_ms(payload, 0);

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

      const auto value_ptr = std::make_shared<const json>(std::move(value));
      const auto snapshot = std::atomic_load(&data_routes_snapshot_);
      if (!snapshot) return;

      main_thread_.post([this, snapshot, subject, value_ptr, ts_ms, meta]() {
        const auto it = snapshot->by_subject.find(subject);
        if (it == snapshot->by_subject.end()) return;
        const auto& routes = it->second;
        if (routes.empty()) return;

        const std::int64_t now = now_ms();
        for (const auto& r : routes) {
          if (!r.buf) continue;
          if (r.timeout_ms > 0 && ts_ms > 0 && (now - ts_ms) > r.timeout_ms) {
            continue;
          }
          auto& buf = *r.buf;
          std::lock_guard<std::mutex> lock(buf.mu);
          buf.last_seen_value = value_ptr;
          buf.last_seen_ts_ms = ts_ms;
          if (buf.strategy == EdgeStrategy::kLatest) {
            buf.queue.clear();
          }
          buf.queue.emplace_back(value_ptr, ts_ms);
        }

        if (cfg_.data_delivery != DataDeliveryMode::kPush && cfg_.data_delivery != DataDeliveryMode::kBoth) {
          return;
        }

        std::vector<DataReceivableNode*> nodes;
        {
          std::lock_guard<std::mutex> lock(handlers_mu_);
          nodes = data_nodes_;
        }
        if (nodes.empty()) return;

        for (const auto& r : routes) {
          if (r.timeout_ms > 0 && ts_ms > 0 && (now - ts_ms) > r.timeout_ms) {
            continue;
          }
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
              n->on_data(r.to_node_id, r.to_port, *value_ptr, ts_ms, m);
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

  data_inputs_ = std::move(next_inputs);
  std::shared_ptr<const _DataRoutingSnapshot> next_snapshot_const = next_snapshot;
  std::atomic_store(&data_routes_snapshot_, std::move(next_snapshot_const));
}

bool ServiceBus::start() {
  stop();

  cfg_.service_id = ensure_token(cfg_.service_id, "service_id");
  if (cfg_.service_name.empty()) {
    cfg_.service_name = cfg_.service_id;
  }
  terminate_.store(false, std::memory_order_release);
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
      ServiceControlPlaneServer::Config{cfg_.service_id, cfg_.nats_url, cfg_.service_name, cfg_.service_class}, &nats_,
      &kv_, this);
  if (!ctrl_->start()) {
    ctrl_.reset();
    kv_.close();
    nats_.close();
    return false;
  }

  load_active_from_kv();

  // Clear any stale ready flag from a previous run as early as possible.
  (void)kv_set_ready(kv_, cfg_.service_id, false, "starting");

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
        const std::int64_t ts_ms = coerce_inbound_ts_ms(payload, 0);
        nlohmann::json meta = payload;
        try {
          if (meta.is_object()) meta.erase("value");
        } catch (...) {
          meta = nlohmann::json::object();
        }

        {
          std::lock_guard<std::mutex> lock(state_mu_);
          state_cache_[{node_id, field}] = {value, ts_ms};
        }

        main_thread_.post([this, node_id, field, value, ts_ms, meta]() {
          bool allow_fanout = true;
          try {
            if (meta.is_object() && meta.contains("_noStateFanout") && meta["_noStateFanout"].is_boolean() &&
                meta["_noStateFanout"].get<bool>()) {
              allow_fanout = false;
            }
          } catch (...) {
            allow_fanout = true;
          }
          deliver_state_local(node_id, field, value, ts_ms, meta, allow_fanout);
        });
      },
      true);

  // Seed identity fields (best-effort). Specs should declare these as readonly.
  try {
    (void)kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "svcId", cfg_.service_id, "system",
                            json{{"builtin", true}}, 0, "system");
  } catch (...) {}

  // Announce readiness after endpoints are up.
  (void)kv_set_ready(kv_, cfg_.service_id, true, "start");
  spdlog::info("service_bus started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void ServiceBus::stop() {
  if (kv_.valid()) {
    (void)kv_set_ready(kv_, cfg_.service_id, false, "stop");
  }

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    for (auto& kv : peer_kv_by_service_id_) {
      try {
        if (kv.second) {
          kv.second->stop_watch();
          kv.second->close();
        }
      } catch (...) {
      }
    }
    peer_kv_by_service_id_.clear();
    cross_state_in_.clear();
    cross_state_targets_.clear();
  }

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
    data_inputs_.clear();
    std::atomic_store(&data_routes_snapshot_, std::shared_ptr<const _DataRoutingSnapshot>{});
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_cache_.clear();
    state_access_.clear();
    intra_state_out_.clear();
    cross_state_in_.clear();
    cross_state_targets_.clear();
    has_rungraph_ = false;
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
  (void)kv_set_node_state(kv_, cfg_.service_id, cfg_.service_id, "active", active, source, extra, 0, "runtime");

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
  if (nodes.empty()) {
    std::string node_id_s;
    try {
      node_id_s = ensure_token(node_id, "node_id");
    } catch (...) {
      error_code = "INVALID_ARGS";
      error_message = "invalid nodeId";
      return false;
    }
    std::string field_s = field;
    field_s.erase(field_s.begin(),
                  std::find_if(field_s.begin(), field_s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    field_s.erase(std::find_if(field_s.rbegin(), field_s.rend(),
                               [](unsigned char ch) { return !std::isspace(ch); })
                      .base(),
                  field_s.end());
    if (field_s.empty()) {
      error_code = "INVALID_ARGS";
      error_message = "field must be non-empty";
      return false;
    }

    std::string access;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      const auto it = state_access_.find({node_id_s, field_s});
      if (it != state_access_.end()) access = it->second;
      if (has_rungraph_ && it == state_access_.end()) {
        error_code = "UNKNOWN_FIELD";
        error_message = "unknown state field";
        return false;
      }
    }
    if (!access.empty() && !state_origin_allows_access("external", access)) {
      error_code = "FORBIDDEN";
      error_message = "state field not writable";
      return false;
    }
    publish_state_local(node_id_s, field_s, value, now_ms(), "endpoint", meta, "external", true, true);
    return true;
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
  error_code.clear();
  error_message.clear();
  try {
    json persisted = graph_obj;
    if (!persisted.contains("meta") || !persisted["meta"].is_object()) {
      persisted["meta"] = json::object();
    }
    persisted["meta"]["ts"] = now_ms();
    apply_rungraph_local(persisted, error_code, error_message);
    if (!error_code.empty()) {
      return false;
    }
    const auto bytes = persisted.dump();
    (void)kv_.put(kv_key_rungraph(), bytes.data(), bytes.size());
  } catch (const std::exception& ex) {
    error_code = "INTERNAL";
    error_message = ex.what();
    return false;
  } catch (...) {
    error_code = "INTERNAL";
    error_message = "unknown error";
    return false;
  }
  std::vector<RungraphHandlerNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = rungraph_nodes_;
  }
  // Best-effort hook calls (mirrors pysdk's rungraph hooks boundary).
  for (auto* n : nodes) {
    if (!n) continue;
    try {
      std::string _code;
      std::string _msg;
      (void)n->on_set_rungraph(graph_obj, meta, _code, _msg);
    } catch (...) {
      continue;
    }
  }
  return true;
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

bool ServiceBus::emit_data(const std::string& from_node_id, const std::string& port_id, const json& value,
                           std::int64_t ts_ms) {
  if (!active()) return false;
  return publish_data(nats_, cfg_.service_id, ensure_token(from_node_id, "from_node_id"), ensure_token(port_id, "port_id"),
                      value, ts_ms);
}

std::optional<json> ServiceBus::pull_data(const std::string& node_id, const std::string& port_id) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string pid = ensure_token(port_id, "port_id");

  std::shared_ptr<_InputBuffer> buf_ptr;
  {
    std::lock_guard<std::mutex> lock(data_mu_);
    const auto it = data_inputs_.find({nid, pid});
    if (it == data_inputs_.end()) {
      return std::nullopt;
    }
    buf_ptr = it->second;
  }
  if (!buf_ptr) return std::nullopt;

  const std::int64_t now = now_ms();
  auto& mut = *buf_ptr;
  std::lock_guard<std::mutex> lock(mut.mu);
  if (mut.timeout_ms > 0 && mut.last_seen_ts_ms > 0 && (now - mut.last_seen_ts_ms) > mut.timeout_ms) {
    return std::nullopt;
  }
  if (mut.strategy == EdgeStrategy::kQueue) {
    if (mut.queue.empty()) return std::nullopt;
    const auto& sample = mut.queue.front();
    json v = sample.first ? *sample.first : json(nullptr);
    mut.queue.pop_front();
    return v;
  }

  // latest
  if (!mut.queue.empty()) {
    const auto& sample = mut.queue.back();
    json v = sample.first ? *sample.first : json(nullptr);
    mut.queue.clear();
    return v;
  }
  if (mut.last_seen_ts_ms <= 0) return std::nullopt;
  if (!mut.last_seen_value) return std::nullopt;
  return *mut.last_seen_value;
}

ServiceBus::StateRead ServiceBus::get_state(const std::string& node_id, const std::string& field) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string f = field;
  if (f.empty()) {
    return StateRead{false, json(nullptr), std::nullopt};
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = state_cache_.find({nid, f});
    if (it != state_cache_.end()) {
      return StateRead{true, it->second.first, it->second.second};
    }
  }

  const auto key = kv_key_node_state(nid, f);
  auto raw = kv_.get(key);
  if (!raw.has_value()) {
    return StateRead{false, json(nullptr), std::nullopt};
  }
  json payload = json::object();
  try {
    payload = json::parse(std::string(reinterpret_cast<const char*>(raw->data()), raw->size()), nullptr, false);
  } catch (...) {
    payload = json::object();
  }
  if (!payload.is_object() || !payload.contains("value")) {
    return StateRead{true, json::binary(*raw), std::int64_t{0}};
  }
  const json v = payload["value"];
  const std::int64_t ts = coerce_inbound_ts_ms(payload, 0);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_cache_[{nid, f}] = {v, ts};
  }
  return StateRead{true, v, ts};
}

void ServiceBus::apply_rungraph_local(const json& graph_obj, std::string& error_code, std::string& error_message) {
  using namespace f8::cppsdk::generated;

  F8RuntimeGraph graph{};
  ParseError perr{};
  if (!parse_F8RuntimeGraph(graph_obj, graph, perr)) {
    error_code = "INVALID_RUNGRAPH";
    error_message = perr.message.empty() ? "invalid rungraph" : perr.message;
    return;
  }
  try {
    validate_state_edges_or_throw(graph);
  } catch (const std::exception& ex) {
    error_code = "INVALID_RUNGRAPH";
    error_message = ex.what();
    return;
  }

  // Service/container nodes require nodeId == serviceId.
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (!n.operatorClass.has_value() && n.nodeId != n.serviceId) {
      error_code = "INVALID_RUNGRAPH";
      error_message = "service node requires nodeId == serviceId";
      return;
    }
  }

  std::unordered_map<_NodeFieldKey, std::string, _NodeFieldKeyHash> state_access;
  std::unordered_map<_NodeFieldKey, std::vector<_NodeFieldKey>, _NodeFieldKeyHash> intra_state_out;
  std::unordered_map<_RemoteStateKey, std::vector<_NodeFieldKey>, _RemoteStateKeyHash> cross_state_in;
  std::unordered_set<_NodeFieldKey, _NodeFieldKeyHash> cross_state_targets;
  std::vector<std::tuple<std::string, std::string, std::string>> cross_state_initial_reads;

  const std::string sid = cfg_.service_id;

  // Build access map and validate rungraph-provided stateValues.
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (n.serviceId != sid) continue;
    if (n.nodeId.empty()) {
      error_code = "INVALID_RUNGRAPH";
      error_message = "missing nodeId";
      return;
    }
    std::unordered_map<std::string, std::string> access_by_name;
    if (n.stateFields.has_value()) {
      for (const auto& sf : n.stateFields.value()) {
        const std::string name = sf.name;
        if (name.empty()) continue;
        const std::string access_s = access_to_string(sf.access);
        state_access[{n.nodeId, name}] = access_s;
        access_by_name[name] = access_s;
      }
    }
    if (n.stateValues.is_object()) {
      for (auto it = n.stateValues.begin(); it != n.stateValues.end(); ++it) {
        const std::string k = it.key();
        const auto a_it = access_by_name.find(k);
        if (a_it == access_by_name.end()) {
          error_code = "INVALID_RUNGRAPH";
          error_message = "unknown state value: " + n.nodeId + "." + k;
          return;
        }
        if (a_it->second == "ro") {
          error_code = "INVALID_RUNGRAPH";
          error_message = "read-only state cannot be set by rungraph: " + n.nodeId + "." + k;
          return;
        }
      }
    }
  }

  // Build intra-service state edge fanout table.
  for (const auto& e : graph.edges.value_or(std::vector<F8Edge>{})) {
    if (e.kind != F8EdgeKindEnum::state) continue;
    const std::string from_sid = e.fromServiceId;
    const std::string to_sid = e.toServiceId;
    if (to_sid != sid) continue;

    std::string from_node = e.fromOperatorId.value_or("");
    if (from_node.empty()) from_node = from_sid;
    std::string to_node = e.toOperatorId.value_or("");
    if (to_node.empty()) to_node = sid;
    const std::string from_field = e.fromPort;
    const std::string to_field = e.toPort;
    if (from_sid.empty() || to_node.empty() || from_node.empty() || from_field.empty() || to_field.empty()) continue;

    // Pre-filter to only writable targets for external propagation to reduce per-update overhead.
    {
      const auto it_access = state_access.find({to_node, to_field});
      if (it_access == state_access.end()) continue;
      if (it_access->second == "ro") continue;
    }

    if (from_sid == sid) {
      // Intra-service state edge.
      intra_state_out[{from_node, from_field}].push_back({to_node, to_field});
    } else {
      // Cross-service state binding (remote KV -> local field).
      cross_state_in[{from_sid, from_node, from_field}].push_back({to_node, to_field});
      cross_state_targets.insert({to_node, to_field});
      cross_state_initial_reads.emplace_back(from_sid, from_node, from_field);
    }
  }

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_access_ = std::move(state_access);
    intra_state_out_ = std::move(intra_state_out);
    cross_state_in_ = std::move(cross_state_in);
    cross_state_targets_ = std::move(cross_state_targets);
    has_rungraph_ = true;
  }

  apply_data_routes_from_rungraph(graph_obj);

  // Ensure peer KV watches are running for any cross-state dependencies.
  // This mirrors f8pysdk's cross-service state routing (remote KV watch + initial sync).
  std::unordered_set<std::string> want_peers;
  for (const auto& t : cross_state_initial_reads) {
    want_peers.insert(std::get<0>(t));
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    for (auto it = peer_kv_by_service_id_.begin(); it != peer_kv_by_service_id_.end();) {
      if (want_peers.find(it->first) != want_peers.end()) {
        ++it;
        continue;
      }
      try {
        if (it->second) {
          it->second->stop_watch();
          it->second->close();
        }
      } catch (...) {
      }
      it = peer_kv_by_service_id_.erase(it);
    }
  }

  for (const auto& peer : want_peers) {
    bool has_peer = false;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      has_peer = (peer_kv_by_service_id_.find(peer) != peer_kv_by_service_id_.end());
    }
    if (has_peer) continue;

    auto kv_peer = std::make_unique<KvStore>();
    KvConfig kvc;
    kvc.bucket = kv_bucket_for_service(peer);
    kvc.memory_storage = true;
    kvc.history = 1;
    if (!kv_peer->open_or_create(nats_.jetstream(), kvc)) {
      spdlog::warn("peer KV open failed bucket={} peer={}", kvc.bucket, peer);
      continue;
    }

    const bool ok = kv_peer->watch(
        "nodes.>",
        [this, peer](const std::string& key, const std::vector<std::uint8_t>& bytes) {
          constexpr const char* kPrefix = "nodes.";
          constexpr const char* kStateMarker = ".state.";
          if (key.rfind(kPrefix, 0) != 0) return;
          const std::size_t marker = key.find(kStateMarker);
          if (marker == std::string::npos) return;

          const std::size_t node_begin = std::strlen(kPrefix);
          const std::size_t node_end = marker;
          if (node_end <= node_begin) return;
          const std::string remote_node_id = key.substr(node_begin, node_end - node_begin);

          const std::size_t field_begin = marker + std::strlen(kStateMarker);
          if (field_begin >= key.size()) return;
          const std::string remote_field = key.substr(field_begin);

          std::vector<_NodeFieldKey> targets;
          {
            std::lock_guard<std::mutex> lock(state_mu_);
            const auto it = cross_state_in_.find(_RemoteStateKey{peer, remote_node_id, remote_field});
            if (it == cross_state_in_.end()) return;
            targets = it->second;
          }
          if (targets.empty()) return;

          nlohmann::json payload = nlohmann::json::object();
          try {
            const std::string s(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            payload = nlohmann::json::parse(s, nullptr, false);
          } catch (...) {
            return;
          }
          if (!payload.is_object()) return;

          const nlohmann::json value = payload.contains("value") ? payload["value"] : nlohmann::json();
          const std::int64_t ts_ms = coerce_inbound_ts_ms(payload, now_ms());
          nlohmann::json meta = payload;
          try {
            if (meta.is_object()) meta.erase("value");
          } catch (...) {
            meta = nlohmann::json::object();
          }
          meta["peerServiceId"] = peer;
          meta["remoteKey"] = key;
          meta["fromNodeId"] = remote_node_id;
          meta["fromField"] = remote_field;

          if (state_debug_enabled()) {
            try {
              std::string v_s;
              try {
                v_s = value.dump();
              } catch (...) {
                v_s = "<non_json>";
              }
              if (v_s.size() > 160) v_s = v_s.substr(0, 157) + "...";
              spdlog::info("state_debug[{}] cross_state_watch peer={} key={} ts={} targets={} value={}", cfg_.service_id,
                           peer, key, ts_ms, targets.size(), v_s);
            } catch (...) {
            }
          }

          for (const auto& t : targets) {
            publish_state_local(t.node_id, t.field, value, ts_ms, "state_edge_cross", meta, "external", true, true);
          }
        },
        true);

    if (!ok) {
      kv_peer->close();
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      peer_kv_by_service_id_[peer] = std::move(kv_peer);
    }
    if (state_debug_enabled()) {
      spdlog::info("state_debug[{}] cross_state_watch_started peer={} bucket={}", cfg_.service_id, peer, kvc.bucket);
    }
  }

  // Initial sync for cross-state targets: best-effort pull current remote values once.
  for (const auto& t : cross_state_initial_reads) {
    const std::string peer = std::get<0>(t);
    const std::string remote_node_id = std::get<1>(t);
    const std::string remote_field = std::get<2>(t);
    std::unique_ptr<KvStore>* kv_peer_ptr = nullptr;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      auto it = peer_kv_by_service_id_.find(peer);
      if (it != peer_kv_by_service_id_.end()) kv_peer_ptr = &it->second;
    }
    if (kv_peer_ptr == nullptr || kv_peer_ptr->get() == nullptr) continue;

    std::string remote_key;
    try {
      remote_key = kv_key_node_state(remote_node_id, remote_field);
    } catch (...) {
      continue;
    }
    const auto raw = (*kv_peer_ptr)->get(remote_key);
    if (!raw.has_value()) {
      if (state_debug_enabled()) {
        spdlog::info("state_debug[{}] cross_state_initial_miss peer={} key={}", cfg_.service_id, peer, remote_key);
      }
      continue;
    }

    // Reuse the same parsing path as the watcher callback.
    constexpr const char* kPrefix = "nodes.";
    constexpr const char* kStateMarker = ".state.";
    if (remote_key.rfind(kPrefix, 0) != 0) continue;
    const std::size_t marker = remote_key.find(kStateMarker);
    if (marker == std::string::npos) continue;

    nlohmann::json payload = nlohmann::json::object();
    try {
      const std::string s(reinterpret_cast<const char*>(raw->data()), raw->size());
      payload = nlohmann::json::parse(s, nullptr, false);
    } catch (...) {
      continue;
    }
    if (!payload.is_object()) continue;

    const nlohmann::json value = payload.contains("value") ? payload["value"] : nlohmann::json();
    const std::int64_t ts_ms = coerce_inbound_ts_ms(payload, now_ms());
    nlohmann::json meta = payload;
    try {
      if (meta.is_object()) meta.erase("value");
    } catch (...) {
      meta = nlohmann::json::object();
    }
    meta["peerServiceId"] = peer;
    meta["remoteKey"] = remote_key;
    meta["fromNodeId"] = remote_node_id;
    meta["fromField"] = remote_field;

    std::vector<_NodeFieldKey> targets;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      const auto it = cross_state_in_.find(_RemoteStateKey{peer, remote_node_id, remote_field});
      if (it != cross_state_in_.end()) targets = it->second;
    }
    if (state_debug_enabled()) {
      try {
        std::string v_s;
        try {
          v_s = value.dump();
        } catch (...) {
          v_s = "<non_json>";
        }
        if (v_s.size() > 160) v_s = v_s.substr(0, 157) + "...";
        spdlog::info("state_debug[{}] cross_state_initial_hit peer={} key={} ts={} targets={} value={}", cfg_.service_id,
                     peer, remote_key, ts_ms, targets.size(), v_s);
      } catch (...) {
      }
    }
    for (const auto& tgt : targets) {
      publish_state_local(tgt.node_id, tgt.field, value, ts_ms, "state_edge_cross", meta, "external", true, true);
    }
  }

  // Apply per-node stateValues (best-effort reconcile using rungraph meta.ts).
  std::int64_t rungraph_ts = 0;
  try {
    if (graph.meta.has_value()) rungraph_ts = graph.meta->ts.value_or(0);
  } catch (...) {
    rungraph_ts = 0;
  }

  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (n.serviceId != sid) continue;
    const std::string node_id = n.nodeId;
    if (!n.stateValues.is_object()) continue;
    for (auto it = n.stateValues.begin(); it != n.stateValues.end(); ++it) {
      const std::string field = it.key();
      const json v = it.value();

      // Cross-service state edges are directional: downstream follows upstream.
      // Do not apply rungraph stateValues to fields that are cross-state targets,
      // otherwise local UI defaults (often empty) can clobber remote-propagated values.
      if (cross_state_targets_.find(_NodeFieldKey{node_id, field}) != cross_state_targets_.end()) {
        if (state_debug_enabled()) {
          spdlog::info("state_debug[{}] cross_state_skip_rungraph node={}.{} value={}", cfg_.service_id, node_id, field,
                       v.dump());
        }
        continue;
      }

      if (rungraph_ts > 0) {
        const auto existing = get_state(node_id, field);
        if (existing.found) {
          try {
            if (existing.value == v) {
              continue;
            }
          } catch (...) {
          }
          if (existing.ts_ms.has_value() && existing.ts_ms.value() >= rungraph_ts) {
            continue;
          }
        }
      }
      publish_state_local(node_id, field, v, rungraph_ts > 0 ? rungraph_ts : now_ms(), "rungraph",
                          json{{"via", "rungraph"}, {"rungraphReconcile", true}, {"_noStateFanout", true}}, "rungraph",
                          true, false);
    }
  }

  // Seed identity fields (`svcId`, `operatorId`) when declared in stateFields.
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (n.serviceId != sid) continue;
    const std::string node_id = n.nodeId;
    bool has_svc_id = false;
    bool has_operator_id = false;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      has_svc_id = state_access_.find({node_id, "svcId"}) != state_access_.end();
      has_operator_id = n.operatorClass.has_value() && state_access_.find({node_id, "operatorId"}) != state_access_.end();
    }
    if (has_svc_id) {
      publish_state_local(node_id, "svcId", n.serviceId.empty() ? sid : n.serviceId, rungraph_ts > 0 ? rungraph_ts : now_ms(),
                          "system", json{{"builtin", true}, {"_noStateFanout", true}}, "system", false, false);
    }
    if (has_operator_id) {
      publish_state_local(node_id, "operatorId", n.nodeId, rungraph_ts > 0 ? rungraph_ts : now_ms(), "system",
                          json{{"builtin", true}, {"_noStateFanout", true}}, "system", false, false);
    }
  }
}

void ServiceBus::publish_state_local(const std::string& node_id, const std::string& field, const json& value,
                                     std::int64_t ts_ms, const std::string& source, const json& meta,
                                     const std::string& origin, bool deliver_local, bool allow_state_fanout) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string f = field;
  if (f.empty()) return;

  // Value-dedupe.
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = state_cache_.find({nid, f});
    if (it != state_cache_.end()) {
      try {
        if (it->second.first == value) {
          return;
        }
      } catch (...) {
      }
    }
  }

  // Access enforcement when rungraph is known.
  std::string access;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = state_access_.find({nid, f});
    if (it != state_access_.end()) access = it->second;
    if (has_rungraph_ && it == state_access_.end()) {
      return;
    }
  }
  if (!access.empty() && !state_origin_allows_access(origin, access)) {
    return;
  }

  const json extra = meta.is_object() ? meta : json::object();
  (void)kv_set_node_state(kv_, cfg_.service_id, nid, f, value, source, extra, ts_ms, origin);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_cache_[{nid, f}] = {value, ts_ms};
  }
  if (deliver_local) {
    main_thread_.post([this, nid, f, value, ts_ms, extra, allow_state_fanout]() {
      deliver_state_local(nid, f, value, ts_ms, extra, allow_state_fanout);
    });
  }
}

void ServiceBus::deliver_state_local(const std::string& node_id, const std::string& field, const json& value,
                                     std::int64_t ts_ms, const json& meta, bool allow_state_fanout) {
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
  if (allow_state_fanout) {
    route_intra_state_edges(node_id, field, value, ts_ms);
  }
}

void ServiceBus::route_intra_state_edges(const std::string& from_node_id, const std::string& from_field,
                                         const json& value, std::int64_t ts_ms) {
  std::vector<_NodeFieldKey> targets;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = intra_state_out_.find({from_node_id, from_field});
    if (it != intra_state_out_.end()) {
      targets = it->second;
    }
  }
  if (targets.empty()) return;
  for (const auto& t : targets) {
    publish_state_local(t.node_id, t.field, value, ts_ms, "state_edge_intra", json{{"fromNodeId", from_node_id}, {"fromField", from_field}},
                        "external", true, true);
  }
}

}  // namespace f8::cppsdk
