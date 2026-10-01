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
class TempestNode final : public OperatorNode, public ComputableNode {
 public:
  TempestNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"phase", "amp", "phaseOffset", "eccentric", "dc"}),
                     data_port_names(node.dataOutPorts, {"out"}),
                     state_names(node.stateFields, {"dc", "amp", "phaseOffset", "eccentric"}),
                     strings_or(node.execInPorts, {}), strings_or(node.execOutPorts, {})) {
    dc_ = json_number_or(initial_state.value("dc", 0.5), 0.5);
    amp_ = json_number_or(initial_state.value("amp", 0.5), 0.5);
    phase_offset_ = json_number_or(initial_state.value("phaseOffset", 0.0), 0.0);
    eccentric_ = json_number_or(initial_state.value("eccentric", 0.0), 0.0);
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "dc") dc_ = json_number_or(value, dc_);
    if (field == "amp") amp_ = json_number_or(value, amp_);
    if (field == "phaseOffset") phase_offset_ = json_number_or(value, phase_offset_);
    if (field == "eccentric") eccentric_ = json_number_or(value, eccentric_);
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const json out = compute_output("out", ts_ms);
    if (!out.is_null()) (void)emit("out", out, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "out") return nullptr;
    const double phase = std::fmod(std::fmod(json_number_or(pull("phase", ctx_id).value_or(0.0), 0.0), 1.0) + 1.0, 1.0);
    const double amp = json_number_or(pull("amp", ctx_id).value_or(amp_), amp_);
    const double offset = json_number_or(pull("phaseOffset", ctx_id).value_or(phase_offset_), phase_offset_);
    const double eccentric = json_number_or(pull("eccentric", ctx_id).value_or(eccentric_), eccentric_);
    const double dc = json_number_or(pull("dc", ctx_id).value_or(dc_), dc_);
    const double theta = 2.0 * kPi * (phase + offset);
    return amp * std::cos(theta + eccentric * std::sin(theta)) + dc;
  }

 private:
  double dc_ = 0.5;
  double amp_ = 0.5;
  double phase_offset_ = 0.0;
  double eccentric_ = 0.0;
};

json tempest_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.tempest"},
              {"version", "0.0.1"},
              {"label", "Tempest"},
              {"description", "Tempest phase transform (phase-modulated cosine)."},
              {"tags", json::array({"signal", "waveform", "generator", "oscillator", "tempest"})},
              {"dataInPorts",
               json::array({data_port("phase", "Phase input (0..1).", number_schema(), true, true),
                            data_port("amp", "Amplitude override.", number_schema(), false, false),
                            data_port("phaseOffset", "Phase offset override (0..1).", number_schema(), false, false),
                            data_port("eccentric", "Eccentricity override", number_schema(), false, false),
                            data_port("dc", "DC offset override.", number_schema(), false, false)})},
              {"dataOutPorts", json::array({data_port("out", "tempest output", number_schema(), false, true)})},
              {"stateFields",
               json::array({state_field("dc", "DC", "Default DC offset.", number_schema(0.5)),
                            state_field("amp", "Amp", "Default amplitude.", number_schema(0.5)),
                            state_field("phaseOffset", "Phase Offset", "Fraction of a full cycle added to the phase.",
                                        number_schema(0.0, 0.0, 1.0)),
                            state_field("eccentric", "Eccentric", "Controls curvature of the inner sine.", number_schema(0.0))})}};
}

}  // namespace

void register_tempest_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(tempest_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.tempest",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<TempestNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
