#include "demo_service.h"

#include <utility>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/data_bus.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"

namespace f8::sdk_demo {

using json = nlohmann::json;

DemoService::DemoService(Config cfg) : cfg_(std::move(cfg)) {}

DemoService::~DemoService() { stop(); }

bool DemoService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(f8::cppsdk::ServiceBus::Config{cfg_.service_id, cfg_.nats_url, true});
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);
  bus_->add_set_state_node(this);
  bus_->add_rungraph_node(this);
  bus_->add_command_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  // Seed a few demo state fields.
  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("active", active_.load(std::memory_order_acquire), "init", json::object());
  publish_state_if_changed("echo", "", "init", json::object());
  publish_state_if_changed("lastData", json::object(), "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("sdk_demo started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void DemoService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;

  if (bus_) {
    bus_->stop();
  }
  bus_.reset();
}

void DemoService::tick() {
  if (!running()) return;

  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
  }

  // Periodic demo heartbeat published on the data bus (consumable by other services via rungraph data edges).
  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_heartbeat_ms_ == 0 || now - last_heartbeat_ms_ >= 1000) {
    last_heartbeat_ms_ = now;
    ++heartbeat_seq_;
    if (bus_) {
      (void)f8::cppsdk::publish_data(bus_->nats(), cfg_.service_id, cfg_.service_id, "heartbeat",
                                    json{{"seq", heartbeat_seq_}, {"ts", now}}, now);
    }
  }
}

void DemoService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                           const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) {
    (void)f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, value, source, meta);
  }
}

void DemoService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  publish_state_if_changed("active", active, "lifecycle", meta);
  spdlog::info("sdk_demo lifecycle active={} meta={}", active, meta.dump());
}

void DemoService::on_state(const std::string& node_id, const std::string& field, const json& value, std::int64_t ts_ms,
                           const json& meta) {
  (void)ts_ms;
  // Only react to our own service node state updates (from external actors).
  if (node_id != cfg_.service_id) return;
  std::string ec, em;
  (void)on_set_state(node_id, field, value, meta, ec, em);
}

void DemoService::on_data(const std::string& node_id, const std::string& port, const json& value, std::int64_t ts_ms,
                          const json& meta) {
  // Demonstrate receiving cross-service data edges routed by the ServiceBus rungraph.
  json last = json::object({{"nodeId", node_id}, {"port", port}, {"value", value}, {"ts", ts_ms}, {"meta", meta}});
  publish_state_if_changed("lastData", last, "data", json::object());
  spdlog::info("sdk_demo on_data {}.{} value={} meta={}", node_id, port, value.dump(), meta.dump());
}

bool DemoService::on_set_state(const std::string& node_id, const std::string& field, const json& value, const json& meta,
                               std::string& error_code, std::string& error_message) {
  error_code.clear();
  error_message.clear();
  if (node_id != cfg_.service_id) {
    error_code = "INVALID_ARGS";
    error_message = "nodeId must equal serviceId";
    return false;
  }
  const std::string f = field;
  if (f == "active") {
    if (!value.is_boolean()) {
      error_code = "INVALID_VALUE";
      error_message = "active must be boolean";
      return false;
    }
    const bool a = value.get<bool>();
    active_.store(a, std::memory_order_release);
    publish_state_if_changed("active", a, "endpoint", meta);
    return true;
  }
  if (f == "echo") {
    if (!value.is_string()) {
      error_code = "INVALID_VALUE";
      error_message = "echo must be string";
      return false;
    }
    publish_state_if_changed("echo", value.get<std::string>(), "endpoint", meta);
    return true;
  }
  error_code = "UNKNOWN_FIELD";
  error_message = "unknown field";
  return false;
}

bool DemoService::on_set_rungraph(const json& graph_obj, const json& meta, std::string& error_code,
                                  std::string& error_message) {
  (void)graph_obj;
  (void)meta;
  error_code.clear();
  error_message.clear();
  // The ServiceBus already uses this rungraph to subscribe data subjects; this hook is for user services.
  return true;
}

bool DemoService::on_command(const std::string& call, const json& args, const json& meta, json& result,
                             std::string& error_code, std::string& error_message) {
  (void)meta;
  error_code.clear();
  error_message.clear();
  result = json::object();

  if (call == "ping") {
    result["pong"] = true;
    return true;
  }
  if (call == "echo") {
    if (!args.is_object() || !args.contains("text") || !args["text"].is_string()) {
      error_code = "INVALID_ARGS";
      error_message = "echo expects args: {\"text\": string}";
      return false;
    }
    const std::string text = args["text"].get<std::string>();
    publish_state_if_changed("echo", text, "cmd", json::object());
    result["echo"] = text;
    return true;
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

}  // namespace f8::sdk_demo

