#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

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

  struct Config {
    std::string service_id;
    std::string nats_url = "nats://127.0.0.1:4222";
    bool kv_memory_storage = true;
  };

  using LifecycleCallback = std::function<void(bool active, const json& meta)>;
  using SetStateHandler =
      std::function<bool(const std::string& node_id, const std::string& field, const json& value, const json& meta,
                         std::string& error_code, std::string& error_message)>;
  using SetRungraphHandler =
      std::function<bool(const json& graph_obj, const json& meta, std::string& error_code, std::string& error_message)>;
  using CommandHandler = std::function<bool(const std::string& call, const json& args, const json& meta, json& result,
                                            std::string& error_code, std::string& error_message)>;

  explicit ServiceBus(Config cfg);
  ~ServiceBus();
  ServiceBus(const ServiceBus&) = delete;
  ServiceBus& operator=(const ServiceBus&) = delete;
  ServiceBus(ServiceBus&&) = delete;
  ServiceBus& operator=(ServiceBus&&) = delete;

  // Handlers are optional; unhandled calls will be rejected.
  void set_lifecycle_callback(LifecycleCallback cb);
  void set_state_handler(SetStateHandler cb);
  void set_rungraph_handler(SetRungraphHandler cb);
  void set_command_handler(CommandHandler cb);

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

  // Apply lifecycle locally and persist to KV (best-effort).
  void set_active_local(bool active, const json& meta, const std::string& source = "cmd");

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

  Config cfg_;
  std::atomic<bool> active_{true};
  std::atomic<bool> terminate_{false};

  mutable std::mutex term_mu_;
  std::condition_variable term_cv_;

  NatsClient nats_;
  KvStore kv_;
  std::unique_ptr<ServiceControlPlaneServer> ctrl_;

  LifecycleCallback on_lifecycle_;
  SetStateHandler on_set_state_;
  SetRungraphHandler on_set_rungraph_;
  CommandHandler on_command_;
};

}  // namespace f8::cppsdk
