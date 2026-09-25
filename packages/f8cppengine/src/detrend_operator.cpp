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
class DetrendNode final : public OperatorNode, public ComputableNode {
 public:
  DetrendNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"value"}), data_port_names(node.dataOutPorts, {"value"}),
                     state_names(node.stateFields, {"mode", "alpha", "reset_on_state_change"}),
                     strings_or(node.execInPorts, {}), strings_or(node.execOutPorts, {})) {
    mode_ = normalize_mode(initial_state.value("mode", "CONSTANT"));
    alpha_ = clamp_double(json_number_or(initial_state.value("alpha", 0.05), 0.05), 0.0, 1.0);
    reset_on_state_change_ = json_bool_or(initial_state.value("reset_on_state_change", true), true);
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "mode") mode_ = normalize_mode(value);
    if (field == "alpha") alpha_ = clamp_double(json_number_or(value, alpha_), 0.0, 1.0);
    if (field == "reset_on_state_change") reset_on_state_change_ = json_bool_or(value, reset_on_state_change_);
    if (reset_on_state_change_) trackers_.clear();
    dirty_ = true;
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
    const auto raw = pull("value", ctx_id);
    if (!raw.has_value()) return format_number_sequence(last_output_);
    const auto sample = json_number_sequence(raw.value());
    if (sample.empty()) return format_number_sequence(last_output_);
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && sample == last_input_) {
      return format_number_sequence(last_output_);
    }
    if (trackers_.size() != sample.size()) trackers_.assign(sample.size(), TrendTracker{});
    std::vector<double> out;
    out.reserve(sample.size());
    for (std::size_t i = 0; i < sample.size(); ++i) {
      out.push_back(mode_ == "LINEAR" ? trackers_[i].update_linear(sample[i], alpha_) : trackers_[i].update_constant(sample[i], alpha_));
    }
    last_input_ = sample;
    last_output_ = out;
    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return format_number_sequence(last_output_);
  }

 private:
  struct TrendTracker {
    std::optional<double> level;
    double slope = 0.0;
    double update_constant(double value, double alpha) {
      if (!level.has_value()) level = value;
      else level = (1.0 - alpha) * level.value() + alpha * value;
      return value - level.value();
    }
    double update_linear(double value, double alpha) {
      if (!level.has_value()) {
        level = value;
        slope = 0.0;
        return 0.0;
      }
      const double predicted = level.value() + slope;
      const double residual = value - predicted;
      level = predicted + alpha * residual;
      slope += alpha * alpha * residual;
      return value - level.value();
    }
  };

  static std::string normalize_mode(const json& value) {
    std::string text = value.is_string() ? value.get<std::string>() : "CONSTANT";
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return text == "LINEAR" ? "LINEAR" : "CONSTANT";
  }

  std::string mode_ = "CONSTANT";
  double alpha_ = 0.05;
  bool reset_on_state_change_ = true;
  std::vector<TrendTracker> trackers_;
  std::vector<double> last_input_;
  std::vector<double> last_output_;
  std::optional<std::int64_t> last_ctx_id_;
  bool dirty_ = true;
};

json detrend_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.detrend"},
              {"version", "0.0.1"},
              {"label", "Detrend"},
              {"description", "Removes slow baseline or linear trend from scalar or vector inputs."},
              {"tags", json::array({"signal", "detrend", "filter"})},
              {"dataInPorts", json::array({data_port("value", "Value to detrend.", any_schema(), false, true)})},
              {"dataOutPorts", json::array({data_port("value", "Detrended output.", any_schema(), false, true)})},
              {"stateFields",
               json::array({state_field("mode", "Mode", "Detrend mode.",
                                        string_enum_schema("CONSTANT", {"CONSTANT", "LINEAR"}), "rw", true, true),
                            state_field("alpha", "Alpha", "Trend tracking smoothing factor.", number_schema(0.05, 0.0, 1.0),
                                        "rw", true, true, json{{"kind", "slider"}}),
                            state_field("reset_on_state_change", "Reset On State Change",
                                        "Reset tracker history when parameters change.", boolean_schema(true), "rw", true, false)})}};
}

}  // namespace

void register_detrend_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(detrend_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.detrend",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<DetrendNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
