#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/main_thread_queue.h"
#include "f8cppsdk/rungraph_routes.h"
#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/nats_client.h"
#include "f8cppsdk/service_control_plane.h"
#include "f8cppsdk/service_control_plane_server.h"

namespace f8::cppsdk {

// Minimal, protocol-compatible service bus for C++ services.
//
// Goals (phase 1):
// - Own NATS+KV connections and lifecycle state (`active`).
// - Expose built-in micro endpoints via `ServiceControlPlaneServer`.
// - Provide a terminate/quit latch to let services exit gracefully.
// - Keep wire protocol compatible with f8pysdk (KV keys, endpoint subjects, payload schema).
class ServiceBus final : public ServiceControlHandler {
 public:
  using json = nlohmann::json;

  enum class DataDeliveryMode {
    kPull,
    kPush,
    kBoth,
  };

  struct StateRead {
    bool found = false;
    json value = json(nullptr);
    std::optional<std::int64_t> ts_ms;
  };

  struct Config {
    std::string service_id;
    std::string nats_url = "nats://127.0.0.1:4222";
    bool kv_memory_storage = true;
    std::string service_name;
    std::string service_class;
    bool publish_all_data = true;
    DataDeliveryMode data_delivery = DataDeliveryMode::kBoth;
  };

  explicit ServiceBus(Config cfg);
  ~ServiceBus();
  ServiceBus(const ServiceBus&) = delete;
  ServiceBus& operator=(const ServiceBus&) = delete;
  ServiceBus(ServiceBus&&) = delete;
  ServiceBus& operator=(ServiceBus&&) = delete;

  // Handlers are optional; unhandled calls will be rejected.
  void add_lifecycle_node(LifecycleNode* node);
  void add_stateful_node(StatefulNode* node);
  void add_data_node(DataReceivableNode* node);
  void add_set_state_node(SetStateHandlerNode* node);
  void add_rungraph_node(RungraphHandlerNode* node);
  void add_command_node(CommandableNode* node);

  bool start();
  void stop();

  bool active() const { return active_.load(std::memory_order_acquire); }
  bool terminate_requested() const { return terminate_.load(std::memory_order_acquire); }

  // Block until terminate/quit is requested.
  void wait_terminate();

  // Expose underlying transports for high-performance services.
  NatsClient& nats() { return nats_; }
  const NatsClient& nats() const { return nats_; }
  KvStore& kv() { return kv_; }
  const KvStore& kv() const { return kv_; }

  // Pump tasks that must run on the service main/tick thread.
  std::size_t drain_main_thread(std::size_t max_tasks = 0);

  // Apply lifecycle locally and persist to KV (best-effort).
  void set_active_local(bool active, const json& meta, const std::string& source = "cmd");

  // ---- data -----------------------------------------------------------
  // Publish a data sample (wire-compatible with f8pysdk ServiceBus.emit_data).
  bool emit_data(const std::string& from_node_id, const std::string& port_id, const json& value,
                 std::int64_t ts_ms = 0);

  // Pull buffered inbound data for (node,port). Returns nullopt if empty/stale.
  std::optional<json> pull_data(const std::string& node_id, const std::string& port_id);

  // ---- state ----------------------------------------------------------
  // Read state from local cache/KV (wire-compatible with f8pysdk get_state).
  StateRead get_state(const std::string& node_id, const std::string& field);

 // ---- ServiceControlHandler (endpoints) ------------------------------
  bool is_active() const override;
  void on_activate(const json& meta) override;
  void on_deactivate(const json& meta) override;
  void on_set_active(bool active, const json& meta) override;
  bool on_set_state(const std::string& node_id, const std::string& field, const json& value, const json& meta,
                    std::string& error_code, std::string& error_message) override;
  bool on_set_rungraph(const json& graph_obj, const json& meta, std::string& error_code,
                       std::string& error_message) override;
  bool on_command(const std::string& call, const json& args, const json& meta, json& result, std::string& error_code,
                  std::string& error_message) override;

 private:
  void load_active_from_kv();
  void apply_data_routes_from_rungraph(const json& graph_obj);
  void apply_rungraph_local(const json& graph_obj, std::string& error_code, std::string& error_message);
  void publish_state_local(const std::string& node_id, const std::string& field, const json& value, std::int64_t ts_ms,
                           const std::string& source, const json& meta, const std::string& origin,
                           bool deliver_local, bool allow_state_fanout);
  void deliver_state_local(const std::string& node_id, const std::string& field, const json& value, std::int64_t ts_ms,
                           const json& meta, bool allow_state_fanout);
  void route_intra_state_edges(const std::string& from_node_id, const std::string& from_field, const json& value,
                               std::int64_t ts_ms);

  Config cfg_;
  std::atomic<bool> active_{true};
  std::atomic<bool> terminate_{false};

  mutable std::mutex term_mu_;
  std::condition_variable term_cv_;

  NatsClient nats_;
  KvStore kv_;
  std::unique_ptr<ServiceControlPlaneServer> ctrl_;

  MainThreadQueue main_thread_;

  mutable std::mutex lifecycle_mu_;
  std::vector<LifecycleNode*> lifecycle_nodes_;

  mutable std::mutex handlers_mu_;
  std::vector<SetStateHandlerNode*> set_state_nodes_;
  std::vector<RungraphHandlerNode*> rungraph_nodes_;
  std::vector<CommandableNode*> command_nodes_;
  std::vector<StatefulNode*> stateful_nodes_;
  std::vector<DataReceivableNode*> data_nodes_;

  struct _NodePortKey {
    std::string node_id;
    std::string port;
    bool operator==(const _NodePortKey& other) const { return node_id == other.node_id && port == other.port; }
  };
  struct _NodePortKeyHash {
    std::size_t operator()(const _NodePortKey& k) const noexcept {
      return std::hash<std::string>{}(k.node_id) ^ (std::hash<std::string>{}(k.port) << 1);
    }
  };

  struct _NodeFieldKey {
    std::string node_id;
    std::string field;
    bool operator==(const _NodeFieldKey& other) const { return node_id == other.node_id && field == other.field; }
  };
  struct _NodeFieldKeyHash {
    std::size_t operator()(const _NodeFieldKey& k) const noexcept {
      return std::hash<std::string>{}(k.node_id) ^ (std::hash<std::string>{}(k.field) << 1);
    }
  };

  struct _InputBuffer {
    using JsonPtr = std::shared_ptr<const json>;

    mutable std::mutex mu;
    std::deque<std::pair<JsonPtr, std::int64_t>> queue;
    JsonPtr last_seen_value;
    std::int64_t last_seen_ts_ms = 0;
    EdgeStrategy strategy = EdgeStrategy::kLatest;
    std::int64_t timeout_ms = 0;
  };

  mutable std::mutex data_mu_;
  std::unordered_map<std::string, NatsSubscription> data_subs_;
  std::unordered_map<_NodePortKey, std::shared_ptr<_InputBuffer>, _NodePortKeyHash> data_inputs_;

  struct _RouteRuntime {
    std::string to_node_id;
    std::string to_port;
    std::string from_service_id;
    std::string from_node_id;
    std::string from_port;
    EdgeStrategy strategy = EdgeStrategy::kLatest;
    std::int64_t timeout_ms = 0;
    std::shared_ptr<_InputBuffer> buf;
  };

  struct _DataRoutingSnapshot {
    std::unordered_map<std::string, std::vector<_RouteRuntime>> by_subject;
  };

  std::shared_ptr<const _DataRoutingSnapshot> data_routes_snapshot_;

  mutable std::mutex state_mu_;
  std::unordered_map<_NodeFieldKey, std::pair<json, std::int64_t>, _NodeFieldKeyHash> state_cache_;
  std::unordered_map<_NodeFieldKey, std::string, _NodeFieldKeyHash> state_access_;
  std::unordered_map<_NodeFieldKey, std::vector<_NodeFieldKey>, _NodeFieldKeyHash> intra_state_out_;
  bool has_rungraph_ = false;
};

}  // namespace f8::cppsdk
