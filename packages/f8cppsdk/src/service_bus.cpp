#include "f8cppsdk/service_bus.h"

#include <chrono>
#include <utility>

#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/state_kv.h"

namespace f8::cppsdk {

using json = nlohmann::json;

ServiceBus::ServiceBus(Config cfg) : cfg_(std::move(cfg)) {}

ServiceBus::~ServiceBus() {
  stop();
}

void ServiceBus::set_lifecycle_callback(LifecycleCallback cb) {
  on_lifecycle_ = std::move(cb);
}
void ServiceBus::set_state_handler(SetStateHandler cb) {
  on_set_state_ = std::move(cb);
}
void ServiceBus::set_rungraph_handler(SetRungraphHandler cb) {
  on_set_rungraph_ = std::move(cb);
}
void ServiceBus::set_command_handler(CommandHandler cb) {
  on_command_ = std::move(cb);
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

  try {
    if (on_lifecycle_) {
      on_lifecycle_(active, meta);
    }
  } catch (...) {}
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

  if (!on_set_state_) {
    error_code = "NOT_SUPPORTED";
    error_message = "set_state not supported";
    return false;
  }
  return on_set_state_(node_id, field, value, meta, error_code, error_message);
}

bool ServiceBus::on_set_rungraph(const json& graph_obj, const json& meta, std::string& error_code,
                                 std::string& error_message) {
  if (!on_set_rungraph_) {
    // Allow deploy even if unhandled (runtime-only services).
    return true;
  }
  return on_set_rungraph_(graph_obj, meta, error_code, error_message);
}

bool ServiceBus::on_command(const std::string& call, const json& args, const json& meta, json& result,
                            std::string& error_code, std::string& error_message) {
  if (call == "terminate" || call == "quit") {
    spdlog::info("service_bus terminate requested serviceId={}", cfg_.service_id);
    terminate_.store(true, std::memory_order_release);
    term_cv_.notify_all();
    result = json::object();
    result["terminating"] = true;
    // Also forward to the user-level command handler (best-effort). This lets
    // GUI/event-loop services set their own shutdown flags while keeping the
    // bus-level termination latch available for blocking services.
    if (on_command_) {
      try {
        json out;
        std::string ec;
        std::string em;
        (void)on_command_(call, args, meta, out, ec, em);
      } catch (...) {}
    }
    return true;
  }

  if (!on_command_) {
    error_code = "UNKNOWN_CALL";
    error_message = "unknown call: " + call;
    return false;
  }
  return on_command_(call, args, meta, result, error_code, error_message);
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
