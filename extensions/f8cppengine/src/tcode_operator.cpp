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
class TCodeNode final : public OperatorNode, public ComputableNode {
 public:
  TCodeNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, default_axes()), data_port_names(node.dataOutPorts, {"tcode"}),
                     state_names(node.stateFields, {"intervalMs"}), strings_or(node.execInPorts, {}),
                     strings_or(node.execOutPorts, {})) {
    interval_ms_ = std::max(1, js_round(json_number_or(initial_state.value("intervalMs", 20), 20)));
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field != "intervalMs") return value;
    const auto numeric = json_number(value);
    if (!numeric.has_value()) throw std::invalid_argument("intervalMs must be a number");
    const int interval = std::max(1, js_round(numeric.value()));
    if (interval > 50000) throw std::invalid_argument("intervalMs must be <= 50000");
    return interval;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    if (field == "intervalMs") interval_ms_ = validate_state(field, value, ts_ms, meta).get<int>();
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const json out = compute_output("tcode", ts_ms);
    if (!out.is_null()) (void)emit("tcode", out, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "tcode") return nullptr;
    const int interval = std::max(1, js_round(json_number_or(pull("intervalMs", ctx_id).value_or(interval_ms_), interval_ms_)));
    std::vector<std::string> commands;
    for (const auto& axis : axes()) {
      const auto raw = pull(axis, ctx_id);
      if (!raw.has_value()) continue;
      const auto numeric = json_number(raw.value());
      if (!numeric.has_value()) continue;
      const int payload = js_round(clamp_double(numeric.value(), 0.0, 1.0) * 9999.0);
      std::ostringstream part;
      part << axis << std::setw(4) << std::setfill('0') << payload << "I" << std::setw(3) << std::setfill('0') << interval;
      commands.push_back(part.str());
    }
    if (commands.empty()) return "";
    std::ostringstream out;
    for (std::size_t i = 0; i < commands.size(); ++i) {
      if (i > 0) out << ' ';
      out << commands[i];
    }
    out << '\n';
    return out.str();
  }

 private:
  static const std::array<std::string, 10>& axes() {
    static const std::array<std::string, 10> kAxes{"L0", "L1", "L2", "R0", "R1", "R2", "V0", "V1", "A0", "A1"};
    return kAxes;
  }
  static std::vector<std::string> default_axes() {
    std::vector<std::string> out(axes().begin(), axes().end());
    out.push_back("intervalMs");
    return out;
  }

  int interval_ms_ = 20;
};

json tcode_spec() {
  json ins = json::array();
  const std::array<std::string, 10> axes{"L0", "L1", "L2", "R0", "R1", "R2", "V0", "V1", "A0", "A1"};
  for (std::size_t i = 0; i < axes.size(); ++i) {
    ins.push_back(data_port(axes[i], "Axis " + axes[i] + " (0..1).", number_schema(), false, i == 0));
  }
  ins.push_back(data_port("intervalMs", "Optional interval override in milliseconds.", number_schema(20.0, 1.0, 50000.0), false, false));
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".output"},
              {"operatorClass", "f8.tcode"},
              {"version", "0.0.1"},
              {"label", "TCode"},
              {"description", "Generates TCode v0.3 command strings from normalized axis values."},
              {"tags", json::array({"transform", "tcode", "osr", "command", "string"})},
              {"dataInPorts", ins},
              {"dataOutPorts", json::array({data_port("tcode", "TCode v0.3 command string", string_schema(""), false, true)})},
              {"stateFields", json::array({state_field("intervalMs", "Interval (ms)",
                                                       "Default interval appended as I### when intervalMs input is not provided.",
                                                       number_schema(20.0, 1.0, 50000.0), "rw", true, true)})}};
}

}  // namespace

void register_tcode_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(tcode_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.tcode",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<TCodeNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
