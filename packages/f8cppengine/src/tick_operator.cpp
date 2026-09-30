#include "operator_common.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mexce.h>

#include "f8cppengine/constants.h"
#include "f8cppsdk/runtime_node_registry.h"
#include "f8cppsdk/runtime_node.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppengine {

using f8::cppsdk::ComputableNode;
using f8::cppsdk::EntrypointContext;
using f8::cppsdk::EntrypointNode;
using f8::cppsdk::OperatorNode;
using f8::cppsdk::RuntimeNodeRegistry;
using f8::cppsdk::generated::F8RuntimeNode;

namespace {
class TickNode final : public OperatorNode, public EntrypointNode, public ComputableNode {
 public:
  TickNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {}), data_port_names(node.dataOutPorts, {"processingMs", "intervalMs", "latenessMs"}),
                     state_names(node.stateFields, {"tickMs", "hiResTimer"}), strings_or(node.execInPorts, {}),
                     strings_or(node.execOutPorts, {"exec"})) {
    tick_ms_ = coerce_tick_ms(initial_state.value("tickMs", 100));
    hi_res_timer_ = json_bool_or(initial_state.value("hiResTimer", true), true);
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "tickMs") return coerce_tick_ms(value);
    if (field == "hiResTimer") return json_bool_or(value, false);
    return value;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "tickMs") tick_ms_.store(coerce_tick_ms(value), std::memory_order_release);
    if (field == "hiResTimer") hi_res_timer_.store(json_bool_or(value, false), std::memory_order_release);
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)exec_id;
    (void)in_port;
    return exec_out_ports();
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    (void)ctx_id;
    if (port == "intervalMs") return last_interval_ms_.load(std::memory_order_acquire);
    if (port == "latenessMs") return last_lateness_ms_.load(std::memory_order_acquire);
    if (port == "processingMs") return last_processing_ms_.load(std::memory_order_acquire);
    return nullptr;
  }

  void start_entrypoint(const EntrypointContext& ctx) override {
    stop_entrypoint();
    stop_requested_.store(false, std::memory_order_release);
    worker_ = std::thread([this, ctx]() {
      auto next_deadline = std::chrono::steady_clock::now();
      auto last_tick = std::optional<std::chrono::steady_clock::time_point>{};
      while (!stop_requested_.load(std::memory_order_acquire)) {
        const int period_ms = tick_ms_.load(std::memory_order_acquire);
        next_deadline += std::chrono::milliseconds(period_ms);
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_until(lock, next_deadline, [&]() { return stop_requested_.load(std::memory_order_acquire); });
        if (stop_requested_.load(std::memory_order_acquire)) break;
        lock.unlock();

        const auto started = std::chrono::steady_clock::now();
        const std::int64_t exec_id = f8::cppsdk::now_ms();
        std::int64_t interval_ms = 0;
        if (last_tick.has_value()) {
          interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(started - last_tick.value()).count();
        }
        last_tick = started;
        const std::int64_t lateness_ms =
            std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(started - next_deadline).count());
        last_interval_ms_.store(interval_ms, std::memory_order_release);
        last_lateness_ms_.store(lateness_ms, std::memory_order_release);
        (void)emit("intervalMs", interval_ms);
        (void)emit("latenessMs", lateness_ms);
        for (const auto& port : exec_out_ports()) {
          ctx.emit_exec(port, exec_id);
        }
        const std::int64_t processing_ms =
            std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count());
        last_processing_ms_.store(processing_ms, std::memory_order_release);
        (void)emit("processingMs", processing_ms);
        const auto now = std::chrono::steady_clock::now();
        if (next_deadline <= now) {
          const auto missed = std::chrono::duration_cast<std::chrono::milliseconds>(now - next_deadline).count() / period_ms + 1;
          next_deadline += std::chrono::milliseconds(missed * period_ms);
        }
      }
    });
  }

  void stop_entrypoint() override {
    stop_requested_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    stop_requested_.store(false, std::memory_order_release);
  }

 private:
  static int coerce_tick_ms(const json& value) {
    const auto numeric = json_number(value);
    if (!numeric.has_value()) throw std::invalid_argument("tickMs must be an integer");
    const int ms = static_cast<int>(*numeric);
    if (ms < 1) throw std::invalid_argument("tickMs must be >= 1");
    if (ms > 50000) throw std::invalid_argument("tickMs must be <= 50000");
    return ms;
  }

  std::atomic<int> tick_ms_{100};
  std::atomic<bool> hi_res_timer_{true};
  std::atomic<bool> stop_requested_{false};
  std::atomic<std::int64_t> last_interval_ms_{0};
  std::atomic<std::int64_t> last_lateness_ms_{0};
  std::atomic<std::int64_t> last_processing_ms_{0};
  std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
};

json tick_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".execution"},
              {"operatorClass", "f8.tick"},
              {"version", "0.0.1"},
              {"label", "Tick"},
              {"description", "Source operator that generates periodic exec ticks."},
              {"tags", json::array({"execution", "timer", "start", "clock", "entrypoint"})},
              {"stateFields",
               json::array({state_field("tickMs", "Tick (ms)", "Interval in milliseconds for emitting exec ticks.",
                                         integer_schema(100, 1, 50000), "rw", true, true),
                            state_field("hiResTimer", "High-res Timer (Windows)",
                                        "Request high-resolution timer behavior where supported.", boolean_schema(true), "rw", true, false)})},
              {"execOutPorts", json::array({json{{"name", "exec"}}})},
              {"dataOutPorts",
               json::array({data_port("processingMs", "Per-tick processing time in milliseconds.", integer_schema(0), false, false),
                            data_port("intervalMs", "Actual interval between tick starts in milliseconds.", integer_schema(0), false, false),
                            data_port("latenessMs", "How late this tick started relative to its scheduled deadline.", integer_schema(0), false, false)})}};
}

}  // namespace

void register_tick_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(tick_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.tick",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<TickNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
