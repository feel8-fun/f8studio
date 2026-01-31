#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/capabilities.h"
#include "f8cppsdk/service_bus.h"

namespace f8::sdk_demo {

class DemoService final : public f8::cppsdk::LifecycleNode,
                          public f8::cppsdk::StatefulNode,
                          public f8::cppsdk::DataReceivableNode,
                          public f8::cppsdk::SetStateHandlerNode,
                          public f8::cppsdk::RungraphHandlerNode,
                          public f8::cppsdk::CommandableNode {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.sdk_demo";
    std::string nats_url = "nats://127.0.0.1:4222";
  };

  explicit DemoService(Config cfg);
  ~DemoService();

  bool start();
  void stop();
  bool running() const { return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire); }

  void tick();

  // ---- capabilities ---------------------------------------------------
  void on_lifecycle(bool active, const nlohmann::json& meta) override;
  void on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value, std::int64_t ts_ms,
                const nlohmann::json& meta) override;
  void on_data(const std::string& node_id, const std::string& port, const nlohmann::json& value, std::int64_t ts_ms,
               const nlohmann::json& meta) override;
  bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                    const nlohmann::json& meta, std::string& error_code, std::string& error_message) override;
  bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                       std::string& error_message) override;
  bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                  nlohmann::json& result, std::string& error_code, std::string& error_message) override;

 private:
  using json = nlohmann::json;

  void publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                const json& meta);

  Config cfg_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  std::unique_ptr<f8::cppsdk::ServiceBus> bus_;

  std::mutex state_mu_;
  std::unordered_map<std::string, json> published_state_;

  std::int64_t last_heartbeat_ms_ = 0;
  std::uint64_t heartbeat_seq_ = 0;
};

}  // namespace f8::sdk_demo

