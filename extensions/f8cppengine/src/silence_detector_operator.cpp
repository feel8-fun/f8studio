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
class SilenceDetectorNode final : public OperatorNode {
 public:
  SilenceDetectorNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"value"}), data_port_names(node.dataOutPorts, {}),
                     state_names(node.stateFields, {"silenceMs", "deltaThreshold", "isSilent"}),
                     strings_or(node.execInPorts, {"exec"}), strings_or(node.execOutPorts, {"exec"})) {
    silence_ms_ = std::max(0.0, json_number_or(initial_state.value("silenceMs", 500), 500));
    delta_threshold_ = std::max(0.0, json_number_or(initial_state.value("deltaThreshold", 0.001), 0.001));
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "silenceMs") return static_cast<int>(std::max(0.0, json_number_or(value, 0.0)));
    if (field == "deltaThreshold") return std::max(0.0, json_number_or(value, 0.0));
    return value;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    if (field == "silenceMs") silence_ms_ = validate_state(field, value, ts_ms, meta).get<double>();
    if (field == "deltaThreshold") delta_threshold_ = validate_state(field, value, ts_ms, meta).get<double>();
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    sample_and_publish(exec_id);
    return exec_out_ports();
  }

 private:
  void sample_and_publish(std::int64_t ctx_id) {
    const auto raw = pull("value", ctx_id);
    const auto value = raw.has_value() ? json_number(raw.value()) : std::nullopt;
    const double now_s = now_seconds();
    if (value.has_value()) {
      if (!last_value_.has_value() || std::abs(value.value() - last_value_.value()) > delta_threshold_) {
        last_active_s_ = now_s;
      }
      last_value_ = value.value();
    }
    if (!last_active_s_.has_value()) last_active_s_ = now_s;
    const bool next_silent = silence_ms_ > 0.0 && ((now_s - last_active_s_.value()) * 1000.0 >= silence_ms_);
    if (next_silent != is_silent_) {
      is_silent_ = next_silent;
      (void)set_state("isSilent", is_silent_);
    }
  }

  double silence_ms_ = 500.0;
  double delta_threshold_ = 0.001;
  std::optional<double> last_value_;
  std::optional<double> last_active_s_;
  bool is_silent_ = false;
};

json silence_detector_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".analysis"},
              {"operatorClass", "f8.silence_detector"},
              {"version", "0.0.1"},
              {"label", "Silence Detector"},
              {"description", "Detect whether a signal has stayed nearly unchanged for long enough to be considered silent."},
              {"tags", json::array({"analysis", "silence", "activity", "state", "gate"})},
              {"execInPorts", json::array({json{{"name", "exec"}}})},
              {"execOutPorts", json::array({json{{"name", "exec"}}})},
              {"dataInPorts", json::array({data_port("value", "Signal to analyze", number_schema(), false, true)})},
              {"dataOutPorts", json::array()},
              {"stateFields",
               json::array({state_field("silenceMs", "Silence (ms)",
                                        "If the input changes less than deltaThreshold for this long, mark it silent.",
                                        integer_schema(500, 0, 60000), "rw", true, true),
                            state_field("deltaThreshold", "Delta Threshold",
                                        "Absolute change threshold to treat the input as active.", number_schema(0.001, 0.0),
                                        "rw", true, true),
                            state_field("isSilent", "Is Silent",
                                        "Readonly sparse state output indicating whether the signal is currently silent.",
                                        boolean_schema(false), "ro", true, true)})}};
}

}  // namespace

void register_silence_detector_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(silence_detector_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.silence_detector",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<SilenceDetectorNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
