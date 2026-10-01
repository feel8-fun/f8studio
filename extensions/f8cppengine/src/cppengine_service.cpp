#include "f8cppengine/cppengine_service.h"

#include <algorithm>
#include <chrono>
#include <utility>

#include <spdlog/spdlog.h>

#include "f8cppengine/constants.h"
#include "f8cppengine/operators.h"
#include "f8cppsdk/generated/protocol_models.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppengine {

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(const SteadyClock::time_point start, const SteadyClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

CppEngineService::CppEngineService(Config cfg) : cfg_(std::move(cfg)) {
  register_cppengine_specs(registry_);
}

CppEngineService::~CppEngineService() { stop(); }

bool CppEngineService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.service_class = kServiceClass;
  bus_cfg.service_name = "CppEngine";
  bus_cfg.data_delivery = f8::cppsdk::ServiceBus::DataDeliveryMode::kPull;
  bus_cfg.apply_runtime_backend(f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend));

  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  host_ = std::make_unique<f8::cppsdk::ServiceHost>(*bus_, registry_, kServiceClass);
  executor_ = std::make_unique<f8::cppsdk::ExecFlowExecutor>(*bus_);

  host_->start();
  bus_->add_lifecycle_node(this);
  bus_->add_rungraph_node(this);

  if (!bus_->start()) {
    executor_.reset();
    host_.reset();
    bus_.reset();
    return false;
  }

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cppengine started serviceId={}", cfg_.service_id);
  return true;
}

void CppEngineService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;

  clear_auto_samples();
  if (executor_) {
    executor_->set_active(false);
    executor_->clear_nodes();
  }
  if (host_) {
    host_->stop();
  }
  if (bus_) {
    bus_->stop();
  }
  executor_.reset();
  host_.reset();
  bus_.reset();
}

bool CppEngineService::running() const {
  return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
}

void CppEngineService::tick() {
  if (!running()) return;
  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
    }
  }
  process_auto_samples();
}

void CppEngineService::on_lifecycle(bool active, const nlohmann::json& meta) {
  (void)meta;
  active_.store(active, std::memory_order_release);
  if (executor_) {
    executor_->set_active(active);
  }
}

bool CppEngineService::on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta,
                                       std::string& error_code, std::string& error_message) {
  (void)meta;
  error_code.clear();
  error_message.clear();
  if (!host_ || !executor_) {
    error_code = "INTERNAL";
    error_message = "cppengine is not started";
    return false;
  }
  if (!host_->apply_rungraph(graph_obj, error_code, error_message)) {
    return false;
  }
  sync_exec_nodes();
  try {
    executor_->apply_rungraph(graph_obj);
  } catch (const std::exception& exc) {
    error_code = "INVALID_RUNGRAPH";
    error_message = exc.what();
    return false;
  }
  sync_auto_samples(graph_obj);
  return true;
}

nlohmann::json CppEngineService::describe_json() const {
  return registry_.describe(kServiceClass);
}

void CppEngineService::sync_exec_nodes() {
  if (!host_ || !executor_) return;
  executor_->clear_nodes();
  for (f8::cppsdk::OperatorNode* node : host_->operator_nodes()) {
    if (node != nullptr) {
      executor_->register_node(node);
    }
  }
}

void CppEngineService::sync_auto_samples(const nlohmann::json& graph_obj) {
  f8::cppsdk::generated::F8RuntimeGraph graph;
  f8::cppsdk::generated::ParseError parse_error;
  if (!f8::cppsdk::generated::parse_F8RuntimeGraph(graph_obj, graph, parse_error)) {
    clear_auto_samples();
    if (bus_) {
      bus_->report_error(cfg_.service_id, "AUTO_SAMPLE_RUNGRAPH_INVALID",
                         parse_error.message.empty() ? "invalid rungraph" : parse_error.message, "error",
                         "cppengine:auto_sample:rungraph");
    }
    return;
  }

  std::vector<AutoSample> next;
  const std::int64_t now = f8::cppsdk::now_ms();
  for (const auto& service : graph.services.value_or(std::vector<f8::cppsdk::generated::F8RuntimeService>{})) {
    if (service.serviceId != cfg_.service_id) continue;
    for (const auto& request :
         service.autoSampleRequests.value_or(std::vector<f8::cppsdk::generated::F8AutoSampleRequest>{})) {
      if (request.sourceNodeId.empty() || request.sourcePort.empty()) continue;
      AutoSample sample;
      sample.source_node_id = request.sourceNodeId;
      sample.source_port = request.sourcePort;
      sample.interval_ms = std::min<std::int64_t>(5000, std::max<std::int64_t>(8, request.intervalMs));
      sample.next_due_ms = now;
      sample.publish_cross_service = request.publishCrossService.value_or(true);
      next.push_back(std::move(sample));
    }
    break;
  }

  std::lock_guard<std::mutex> lock(auto_samples_mu_);
  auto_samples_ = std::move(next);
  auto_sample_error_last_ms_.clear();
}

void CppEngineService::clear_auto_samples() {
  std::lock_guard<std::mutex> lock(auto_samples_mu_);
  auto_samples_.clear();
  auto_sample_error_last_ms_.clear();
}

void CppEngineService::process_auto_samples() {
  if (!active_.load(std::memory_order_acquire) || !bus_ || !host_) return;
  const std::int64_t now = f8::cppsdk::now_ms();
  std::vector<AutoSample> due;
  {
    std::lock_guard<std::mutex> lock(auto_samples_mu_);
    for (auto& sample : auto_samples_) {
      if (sample.next_due_ms > now) continue;
      due.push_back(sample);
      sample.next_due_ms = now + sample.interval_ms;
    }
  }

  for (const auto& sample : due) {
    if (!sample.publish_cross_service) continue;
    f8::cppsdk::RuntimeNode* runtime_node = host_->get_node(sample.source_node_id);
    if (runtime_node == nullptr) {
      const std::string fingerprint = "cppengine:auto_sample:missing:" + sample.source_node_id + ":" + sample.source_port;
      if (should_report_auto_sample_error(fingerprint, now)) {
        bus_->report_error(sample.source_node_id, "AUTO_SAMPLE_SOURCE_MISSING",
                           "auto sample source node is missing", "warning", fingerprint, now);
      }
      continue;
    }
    auto* computable = dynamic_cast<f8::cppsdk::ComputableNode*>(runtime_node);
    if (computable == nullptr) {
      const std::string fingerprint = "cppengine:auto_sample:not_computable:" + sample.source_node_id + ":" + sample.source_port;
      if (should_report_auto_sample_error(fingerprint, now)) {
        bus_->report_error(sample.source_node_id, "AUTO_SAMPLE_SOURCE_NOT_COMPUTABLE",
                           "auto sample source node does not implement ComputableNode", "warning", fingerprint, now);
      }
      continue;
    }

    const auto start = SteadyClock::now();
    try {
      const nlohmann::json value = computable->compute_output(sample.source_port, now);
      const auto end = SteadyClock::now();
      const double latency_ms = elapsed_ms(start, end);
      bus_->record_monitor_timing("auto_sample", latency_ms, latency_ms, now);
      if (!value.is_null()) {
        (void)bus_->emit_data(sample.source_node_id, sample.source_port, value, now);
      }
    } catch (const std::exception& exc) {
      const auto end = SteadyClock::now();
      const double latency_ms = elapsed_ms(start, end);
      bus_->record_monitor_timing("auto_sample", latency_ms, latency_ms, now);
      const std::string fingerprint = "cppengine:auto_sample:compute:" + sample.source_node_id + ":" + sample.source_port;
      if (should_report_auto_sample_error(fingerprint, now)) {
        bus_->report_error(sample.source_node_id, "AUTO_SAMPLE_COMPUTE_FAILED", exc.what(), "error", fingerprint, now);
      }
    } catch (...) {
      const auto end = SteadyClock::now();
      const double latency_ms = elapsed_ms(start, end);
      bus_->record_monitor_timing("auto_sample", latency_ms, latency_ms, now);
      const std::string fingerprint = "cppengine:auto_sample:compute:" + sample.source_node_id + ":" + sample.source_port;
      if (should_report_auto_sample_error(fingerprint, now)) {
        bus_->report_error(sample.source_node_id, "AUTO_SAMPLE_COMPUTE_FAILED",
                           "auto sample compute_output threw unknown exception", "error", fingerprint, now);
      }
    }
  }
}

bool CppEngineService::should_report_auto_sample_error(const std::string& fingerprint, std::int64_t now_ms) {
  std::lock_guard<std::mutex> lock(auto_samples_mu_);
  const auto it = auto_sample_error_last_ms_.find(fingerprint);
  if (it != auto_sample_error_last_ms_.end() && (now_ms - it->second) < 2000) {
    return false;
  }
  auto_sample_error_last_ms_[fingerprint] = now_ms;
  return true;
}

}  // namespace f8::cppengine
