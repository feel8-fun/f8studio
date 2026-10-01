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
class CosineNode final : public OperatorNode, public ComputableNode {
 public:
  CosineNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"phase", "amp", "dc", "phaseOffset"}),
                     data_port_names(node.dataOutPorts, {"value"}), state_names(node.stateFields, {"dc", "amp", "phaseOffset"}),
                     strings_or(node.execInPorts, {}), strings_or(node.execOutPorts, {})) {
    dc_ = json_number_or(initial_state.value("dc", 0.5), 0.5);
    amp_ = json_number_or(initial_state.value("amp", 0.5), 0.5);
    phase_offset_ = json_number_or(initial_state.value("phaseOffset", 0.0), 0.0);
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "dc") dc_ = json_number_or(value, dc_);
    if (field == "amp") amp_ = json_number_or(value, amp_);
    if (field == "phaseOffset") phase_offset_ = json_number_or(value, phase_offset_);
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const json out = compute_output("value", ts_ms);
    if (!out.is_null()) (void)emit("value", out, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "value") return nullptr;
    const double phase = json_number_or(pull("phase", ctx_id).value_or(0.0), 0.0);
    const double amp = json_number_or(pull("amp", ctx_id).value_or(amp_), amp_);
    const double dc = json_number_or(pull("dc", ctx_id).value_or(dc_), dc_);
    const double offset = json_number_or(pull("phaseOffset", ctx_id).value_or(phase_offset_), phase_offset_);
    return dc + amp * std::cos(2.0 * kPi * (std::fmod(std::fmod(phase, 1.0) + 1.0, 1.0) + offset));
  }

 private:
  double dc_ = 0.5;
  double amp_ = 0.5;
  double phase_offset_ = 0.0;
};

json cosine_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.cosine"},
              {"version", "0.0.1"},
              {"label", "Cosine"},
              {"description", "Cosine phase transform. Provide phase (0..1) from an upstream phase driver."},
              {"tags", json::array({"signal", "cos", "waveform", "generator", "oscillator"})},
              {"dataInPorts",
               json::array({data_port("phase", "Phase input (0..1).", number_schema(), true, true),
                            data_port("amp", "Amplitude override.", number_schema(0.5), false, false),
                            data_port("dc", "DC offset override.", number_schema(0.5), false, false),
                            data_port("phaseOffset", "Phase offset override (0..1).", number_schema(), false, false)})},
              {"dataOutPorts", json::array({data_port("value", "cosine output", number_schema(), false, true)})},
              {"stateFields",
               json::array({state_field("dc", "DC", "Default DC offset.", number_schema(0.5)),
                            state_field("amp", "Amp", "Amplitude.", number_schema(0.5)),
                            state_field("phaseOffset", "Phase Offset", "Normalized phase offset (0.0 to 1.0).",
                                        number_schema(0.0, 0.0, 1.0))})}};
}

}  // namespace

void register_cosine_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(cosine_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.cosine",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<CosineNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
