#include "pending_operator_common.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "operator_common.h"

#include "f8cppengine/constants.h"
#include "f8cppsdk/runtime_node.h"
#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::OperatorNode;
using f8::cppsdk::ComputableNode;
using f8::cppsdk::RuntimeNodeRegistry;
using f8::cppsdk::generated::F8RuntimeNode;

namespace {

class PendingOperatorNode final : public OperatorNode, public ComputableNode {
 public:
  PendingOperatorNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {}), data_port_names(node.dataOutPorts, {}),
                     state_names(node.stateFields, {}), strings_or(node.execInPorts, {}),
                     strings_or(node.execOutPorts, {})),
        operator_class_(node.operatorClass.value_or("unknown")) {
    (void)initial_state;
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)value;
    (void)meta;
    report_error("CPP_OPERATOR_UNIMPLEMENTED",
                 operator_class_ + " is described by f8.cppengine but its native runtime is not implemented yet",
                 "warning", "cpp-unimplemented:" + operator_class_ + ":" + node_id(), ts_ms);
    (void)port;
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)exec_id;
    (void)in_port;
    report_unimplemented();
    return {};
  }

  nlohmann::json compute_output(const std::string& port, std::int64_t ctx_id) override {
    (void)ctx_id;
    if (std::find(data_out_ports().begin(), data_out_ports().end(), port) == data_out_ports().end()) return nullptr;
    report_unimplemented(ctx_id);
    return nullptr;
  }

 private:
  void report_unimplemented(std::int64_t ts_ms = 0) {
    report_error("CPP_OPERATOR_UNIMPLEMENTED",
                 operator_class_ + " is described by f8.cppengine but its native runtime is not implemented yet",
                 "warning", "cpp-unimplemented:" + operator_class_ + ":" + node_id(), ts_ms);
  }

  std::string operator_class_;
};

}  // namespace

json pending_operator_spec(const std::string& operator_class, const std::string& label, const std::string& category,
                           const std::vector<json>& data_in, const std::vector<json>& data_out,
                           const std::vector<json>& states, const std::vector<std::string>& exec_in,
                           const std::vector<std::string>& exec_out, const json& edit_policy) {
  json spec{{"specKind", "operator"},
            {"schemaVersion", "f8operator/1"},
            {"serviceClass", kServiceClass},
            {"paletteCategory", std::string(kServiceClass) + "." + category},
            {"operatorClass", operator_class},
            {"version", "0.0.1"},
            {"label", label},
            {"description", label + " operator described for C++ engine graphs. Native runtime is pending."},
            {"tags", json::array({"cpp", "pending"})}};
  if (!exec_in.empty()) {
    spec["execInPorts"] = json::array();
    for (const auto& name : exec_in) spec["execInPorts"].push_back(json{{"name", name}});
  }
  if (!exec_out.empty()) {
    spec["execOutPorts"] = json::array();
    for (const auto& name : exec_out) spec["execOutPorts"].push_back(json{{"name", name}});
  }
  if (!data_in.empty()) spec["dataInPorts"] = data_in;
  if (!data_out.empty()) spec["dataOutPorts"] = data_out;
  if (!states.empty()) spec["stateFields"] = states;
  if (!edit_policy.is_null()) spec["editPolicy"] = edit_policy;
  return spec;
}

void register_pending_operator_spec(RuntimeNodeRegistry& registry, const json& spec) {
  const std::string operator_class = spec.value("operatorClass", "");
  if (operator_class.empty()) return;
  registry.register_operator_spec(spec, true);
  registry.register_operator_factory(kServiceClass, operator_class,
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<PendingOperatorNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
