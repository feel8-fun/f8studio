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
class PhaseNode final : public OperatorNode, public ComputableNode {
 public:
  PhaseNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"hz", "phase", "reset"}),
                     data_port_names(node.dataOutPorts, {"phase", "phaseTurns"}), state_names(node.stateFields, {"hz"}),
                     strings_or(node.execInPorts, {}), strings_or(node.execOutPorts, {})) {
    hz_ = json_number_or(initial_state.value("hz", 1.0), 1.0);
  }

  void on_lifecycle(bool active, const json& meta) override {
    (void)active;
    (void)meta;
    last_time_s_.reset();
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "hz") hz_ = std::max(0.0, json_number_or(value, hz_));
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const json phase = compute_output("phase", ts_ms);
    const json turns = compute_output("phaseTurns", ts_ms);
    if (!phase.is_null()) (void)emit("phase", phase, ts_ms);
    if (!turns.is_null()) (void)emit("phaseTurns", turns, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "phase" && port != "phaseTurns") return nullptr;
    if (last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && cache_.contains(port)) return cache_[port];
    const auto in_hz = pull("hz", ctx_id);
    const auto in_phase = pull("phase", ctx_id);
    const auto in_reset = pull("reset", ctx_id);
    const double hz = std::max(0.0, in_hz.has_value() ? json_number_or(in_hz.value(), hz_) : hz_);
    if (in_reset.has_value() && json_bool_or(in_reset.value(), false)) turns_ = 0.0;
    if (in_phase.has_value()) {
      const auto phase = json_number(in_phase.value());
      if (phase.has_value()) turns_ = std::floor(turns_) + std::fmod(std::fmod(phase.value(), 1.0) + 1.0, 1.0);
    }
    const double now_s = now_seconds();
    double delta_s = 0.0;
    if (last_time_s_.has_value()) delta_s = std::max(0.0, now_s - last_time_s_.value());
    last_time_s_ = now_s;
    turns_ += hz * delta_s;
    cache_ = json{{"phase", std::fmod(std::fmod(turns_, 1.0) + 1.0, 1.0)}, {"phaseTurns", turns_}};
    last_ctx_id_ = ctx_id;
    return object_value_or_null(cache_, port);
  }

 private:
  double hz_ = 1.0;
  double turns_ = 0.0;
  std::optional<double> last_time_s_;
  std::optional<std::int64_t> last_ctx_id_;
  json cache_ = json::object();
};

json phase_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.phase"},
              {"version", "0.0.1"},
              {"label", "Phase"},
              {"description", "Phase accumulator. Outputs normalized phase (0..1) and unwrapped phase turns."},
              {"tags", json::array({"signal", "phase", "waveform", "generator", "oscillator"})},
              {"dataInPorts",
               json::array({data_port("hz", "Frequency override (Hz).", number_schema(), false, true),
                            data_port("phase", "Absolute phase override (0..1).", number_schema(), false, true),
                            data_port("reset", "If true, reset phase to 0.", boolean_schema(false), false, true)})},
              {"dataOutPorts",
               json::array({data_port("phase", "Normalized phase (0..1).", number_schema(), false, true),
                            data_port("phaseTurns", "Unwrapped phase turns (cycles).", number_schema(), false, true)})},
              {"stateFields", json::array({state_field("hz", "Hz", "Frequency in Hz.", number_schema(1.0, 0.0, 100.0),
                                                       "rw", true, true)})}};
}

}  // namespace

void register_phase_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(phase_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.phase",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<PhaseNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
