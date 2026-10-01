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
class ExecSequenceNode final : public OperatorNode {
 public:
  ExecSequenceNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {}), data_port_names(node.dataOutPorts, {}),
                     state_names(node.stateFields, {}), strings_or(node.execInPorts, {"exec"}),
                     strings_or(node.execOutPorts, {"0", "1", "2"})) {
    (void)initial_state;
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)exec_id;
    (void)in_port;
    return exec_out_ports();
  }
};

json exec_sequence_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".execution"},
              {"operatorClass", "f8.exec_sequence"},
              {"version", "0.0.1"},
              {"label", "Sequence"},
              {"description", "Exec flow splitter: triggers its exec outputs in order."},
              {"tags", json::array({"execution", "flow", "sequence", "branch"})},
              {"execInPorts", json::array({json{{"name", "exec"}}})},
              {"execOutPorts", json::array({json{{"name", "0"}}, json{{"name", "1"}}, json{{"name", "2"}}})},
              {"editPolicy", json{{"execOutPorts", editable_collection_policy()}}}};
}

}  // namespace

void register_exec_sequence_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(exec_sequence_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.exec_sequence",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<ExecSequenceNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
