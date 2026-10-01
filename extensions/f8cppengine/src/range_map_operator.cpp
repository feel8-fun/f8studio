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
class RangeMapNode final : public OperatorNode, public ComputableNode {
 public:
  RangeMapNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"value"}), data_port_names(node.dataOutPorts, {"value"}),
                     state_names(node.stateFields, {"inMin", "inMax", "outMin", "outMax", "curve"}),
                     strings_or(node.execInPorts, {"exec"}), strings_or(node.execOutPorts, {"exec"})) {
    in_min_ = json_number_or(initial_state.value("inMin", 0.0), 0.0);
    in_max_ = json_number_or(initial_state.value("inMax", 1.0), 1.0);
    out_min_ = json_number_or(initial_state.value("outMin", 0.0), 0.0);
    out_max_ = json_number_or(initial_state.value("outMax", 1.0), 1.0);
    curve_ = normalize_curve(initial_state.value("curve", "LINEAR"));
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "inMin") in_min_ = json_number_or(value, in_min_);
    if (field == "inMax") in_max_ = json_number_or(value, in_max_);
    if (field == "outMin") out_min_ = json_number_or(value, out_min_);
    if (field == "outMax") out_max_ = json_number_or(value, out_max_);
    if (field == "curve") curve_ = normalize_curve(value);
    dirty_ = true;
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    const json value = compute_output("value", exec_id);
    if (!value.is_null()) {
      (void)emit("value", value);
    }
    return exec_out_ports();
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)value;
    (void)meta;
    if (port != "value") return;
    const json output = compute_output("value", ts_ms);
    if (!output.is_null()) {
      (void)emit("value", output, ts_ms);
    }
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "value") return nullptr;
    const auto raw = pull("value", ctx_id);
    if (!raw.has_value()) return last_output_.has_value() ? json(last_output_.value()) : json(nullptr);
    const auto numeric = json_number(raw.value());
    if (!numeric.has_value()) return last_output_.has_value() ? json(last_output_.value()) : json(nullptr);
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && last_input_.has_value() &&
        last_input_.value() == numeric.value()) {
      return last_output_.has_value() ? json(last_output_.value()) : json(nullptr);
    }
    double in_min = in_min_;
    double in_max = in_max_;
    double out_min = out_min_;
    double out_max = out_max_;
    if (in_min > in_max) std::swap(in_min, in_max);
    if (out_min > out_max) std::swap(out_min, out_max);
    double output = out_min;
    if ((in_max - in_min) != 0.0) {
      const double clipped = std::min(in_max, std::max(in_min, numeric.value()));
      const double t = apply_curve(curve_, (clipped - in_min) / (in_max - in_min));
      output = out_min + t * (out_max - out_min);
    }
    last_input_ = numeric.value();
    last_output_ = output;
    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return output;
  }

 private:
  double in_min_ = 0.0;
  double in_max_ = 1.0;
  double out_min_ = 0.0;
  double out_max_ = 1.0;
  std::string curve_ = "LINEAR";
  std::optional<double> last_input_;
  std::optional<double> last_output_;
  std::optional<std::int64_t> last_ctx_id_;
  bool dirty_ = true;
};

json range_map_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.range_map"},
              {"version", "0.0.1"},
              {"label", "Range Map"},
              {"description", "Clip input to [inMin,inMax] then remap to [outMin,outMax] with a curve."},
              {"tags", json::array({"map", "range", "normalize", "curve", "transform"})},
              {"dataInPorts", json::array({data_port("value", "Input value.", number_schema(), false, true)})},
              {"dataOutPorts", json::array({data_port("value", "Mapped output.", number_schema(), false, true)})},
              {"stateFields",
               json::array({state_field("inMin", "Input Min", "Input range minimum.", number_schema(0.0)),
                            state_field("inMax", "Input Max", "Input range maximum.", number_schema(1.0)),
                            state_field("outMin", "Output Min", "Output range minimum.", number_schema(0.0), "rw", true, true),
                            state_field("outMax", "Output Max", "Output range maximum.", number_schema(1.0), "rw", true, true),
                            state_field("curve", "Curve", "Mapping curve.",
                                        string_enum_schema("LINEAR", {"LINEAR", "SMOOTHSTEP", "SMOOTHERSTEP", "EASE_IN", "EASE_OUT", "EASE_IN_OUT"}),
                                        "rw", true, true)})}};
}

}  // namespace

void register_range_map_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(range_map_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.range_map",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<RangeMapNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
