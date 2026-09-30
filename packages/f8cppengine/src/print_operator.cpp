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
class PrintNode final : public OperatorNode {
 public:
  PrintNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"value"}), data_port_names(node.dataOutPorts, {}),
                     state_names(node.stateFields, {"strip"}), strings_or(node.execInPorts, {"exec"}),
                     strings_or(node.execOutPorts, {})) {
    strip_ = json_bool_or(initial_state.value("strip", true), true);
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "strip") strip_ = json_bool_or(value, strip_);
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (port == "value") {
      std::cout << "[" << node_id() << "] value=" << json_to_printable(value, strip_) << std::endl;
    }
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    const auto value = pull("value", exec_id);
    std::cout << "[" << node_id() << "] exec=" << exec_id << " value="
              << json_to_printable(value.value_or(nullptr), strip_) << std::endl;
    return {};
  }

 private:
  bool strip_ = true;
};

json print_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".debug"},
              {"operatorClass", "f8.print"},
              {"version", "0.0.1"},
              {"label", "Print"},
              {"description", "Exec/data-driven printer for debugging graph values."},
              {"tags", json::array({"debug", "console", "print"})},
              {"execInPorts", json::array({json{{"name", "exec"}}})},
              {"dataInPorts", json::array({data_port("value", "value to print", any_schema(), false, true)})},
              {"stateFields", json::array({state_field("strip", "Strip",
                                                       "If true, strip whitespace/newlines from string values before printing.",
                                                       boolean_schema(true), "rw", true, false)})}};
}

}  // namespace

void register_print_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(print_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.print",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<PrintNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
