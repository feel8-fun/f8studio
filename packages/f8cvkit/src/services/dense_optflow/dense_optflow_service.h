#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/service_bus.h"

namespace f8::cvkit::dense_optflow {

class DenseOptflowService final : public f8::cppsdk::LifecycleNode,
                                  public f8::cppsdk::StatefulNode,
                                  public f8::cppsdk::DataReceivableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.cvkit.denseoptflow";
    std::string nats_url = "nats://127.0.0.1:4222";
  };

  explicit DenseOptflowService(Config cfg);
  ~DenseOptflowService();

  bool start();
  void stop();
  bool running() const {
    return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
  }

  void tick();

  void on_lifecycle(bool active, const nlohmann::json& meta) override;
  void on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value, std::int64_t ts_ms,
                const nlohmann::json& meta) override;
  void on_data(const std::string& node_id, const std::string& port, const nlohmann::json& value, std::int64_t ts_ms,
               const nlohmann::json& meta) override;

  static nlohmann::json describe();

 private:
  using json = nlohmann::json;

  void publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                const json& meta);
  void handle_request(const json& req, const json& meta);

  Config cfg_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};
  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;
};

}  // namespace f8::cvkit::dense_optflow
