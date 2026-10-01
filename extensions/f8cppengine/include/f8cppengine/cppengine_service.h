#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/exec_flow_executor.h"
#include "f8cppsdk/runtime_backend.h"
#include "f8cppsdk/runtime_node_registry.h"
#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/service_host.h"

namespace f8::cppengine {

class CppEngineService final : public f8::cppsdk::LifecycleNode, public f8::cppsdk::RungraphHandlerNode {
 public:
  struct Config {
    std::string service_id;
    f8::cppsdk::RuntimeBackendConfig runtime_backend;
  };

  explicit CppEngineService(Config cfg);
  ~CppEngineService() override;

  bool start();
  void stop();
  bool running() const;
  void tick();

  void on_lifecycle(bool active, const nlohmann::json& meta) override;
  bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                       std::string& error_message) override;

  nlohmann::json describe_json() const;

 private:
  void sync_exec_nodes();
  void sync_auto_samples(const nlohmann::json& graph_obj);
  void clear_auto_samples();
  void process_auto_samples();
  bool should_report_auto_sample_error(const std::string& fingerprint, std::int64_t now_ms);

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  f8::cppsdk::RuntimeNodeRegistry registry_;
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;
  std::unique_ptr<f8::cppsdk::ServiceHost> host_;
  std::unique_ptr<f8::cppsdk::ExecFlowExecutor> executor_;

  struct AutoSample {
    std::string source_node_id;
    std::string source_port;
    std::int64_t interval_ms = 100;
    std::int64_t next_due_ms = 0;
    bool publish_cross_service = true;
  };

  mutable std::mutex auto_samples_mu_;
  std::vector<AutoSample> auto_samples_;
  std::unordered_map<std::string, std::int64_t> auto_sample_error_last_ms_;
};

}  // namespace f8::cppengine
