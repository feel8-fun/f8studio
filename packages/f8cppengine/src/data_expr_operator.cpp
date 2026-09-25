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
class DataExprNode final : public OperatorNode, public ComputableNode {
 public:
  DataExprNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"x"}), data_port_names(node.dataOutPorts, {"out"}),
                     state_names(node.stateFields, {"code"}), strings_or(node.execInPorts, {"exec"}),
                     strings_or(node.execOutPorts, {"exec"})) {
    code_ = initial_state.value("code", "x");
    rebuild_evaluator_bindings();
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "code") return value.is_string() ? value.get<std::string>() : "";
    return value;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "code") {
      code_ = value.is_string() ? value.get<std::string>() : "";
      evaluator_ready_ = false;
      dirty_ = true;
    }
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    const std::string port = default_output_port();
    const json value = compute_output(port, exec_id);
    if (!value.is_null()) {
      (void)emit(port, value);
    }
    return exec_out_ports();
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const std::string output_port = default_output_port();
    const json output = compute_output(output_port, ts_ms);
    if (!output.is_null()) {
      (void)emit(output_port, output, ts_ms);
    }
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port.empty()) return nullptr;
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && last_outputs_.contains(port)) {
      return last_outputs_[port];
    }
    last_outputs_.clear();
    try {
      bind_inputs(ctx_id);
      compile_if_needed();
      last_outputs_[default_output_port()] = evaluator_.evaluate();
      clear_error("data_expr:" + node_id());
    } catch (const std::exception& exc) {
      report_error("DATA_EXPR_ERROR", exc.what(), "error", "data_expr:" + node_id());
      last_outputs_[default_output_port()] = nullptr;
    }
    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return object_value_or_null(last_outputs_, port);
  }

 private:
  static bool is_bindable_name(const std::string& name) {
    if (name.empty()) return false;
    const unsigned char first = static_cast<unsigned char>(name.front());
    if (!std::isalpha(first) && name.front() != '_') return false;
    for (const char ch : name) {
      const unsigned char uch = static_cast<unsigned char>(ch);
      if (!std::isalnum(uch) && ch != '_') return false;
    }
    return true;
  }

  std::string default_output_port() const {
    for (const auto& port : data_out_ports()) {
      if (port == "out") return port;
    }
    if (!data_out_ports().empty()) return data_out_ports().front();
    return "";
  }

  void rebuild_evaluator_bindings() {
    evaluator_ = mexce::evaluator();
    bound_values_.clear();
    for (const auto& input_port : data_in_ports()) {
      if (!is_bindable_name(input_port)) continue;
      auto inserted = bound_values_.emplace(input_port, 0.0);
      evaluator_.bind(inserted.first->second, input_port);
    }
    evaluator_ready_ = false;
    dirty_ = true;
  }

  void bind_inputs(std::int64_t ctx_id) {
    if (bound_values_.empty() && !data_in_ports().empty()) {
      rebuild_evaluator_bindings();
    }
    for (auto& item : bound_values_) {
      const auto raw = pull(item.first, ctx_id);
      if (!raw.has_value()) continue;
      const auto numeric = json_number(raw.value());
      if (numeric.has_value()) item.second = numeric.value();
    }
  }

  void compile_if_needed() {
    if (evaluator_ready_) return;
    evaluator_.set_expression(code_.empty() ? "0" : code_);
    evaluator_ready_ = true;
  }

  std::string code_ = "x";
  std::optional<std::int64_t> last_ctx_id_;
  json last_outputs_ = json::object();
  std::unordered_map<std::string, double> bound_values_;
  mexce::evaluator evaluator_;
  bool evaluator_ready_ = false;
  bool dirty_ = true;
};

json data_expr_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".expr"},
              {"operatorClass", "f8.data_expr"},
              {"version", "0.0.1"},
              {"label", "Data Expr"},
              {"description", "Evaluate a C++ scalar expression using numeric input values. Python-only syntax and numpy are unsupported."},
              {"tags", json::array({"expr", "math", "logic", "transform", "lightweight"})},
              {"dataInPorts", json::array({data_port("x", "Input value for the expression.", any_schema(), false, true)})},
              {"dataOutPorts", json::array({data_port("out", "Expression result.", any_schema(), false, true)})},
              {"editPolicy", json{{"dataInPorts", editable_collection_policy()}, {"dataOutPorts", editable_collection_policy()}}},
              {"stateFields", json::array({state_field("code", "Expr", "Scalar expression. Reference numeric input port names directly.",
                                                      string_schema("x"), "rw", true, true,
                                                      json{{"kind", "textarea"}, {"language", "cpp"}})})}};
}

}  // namespace

void register_data_expr_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(data_expr_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.data_expr",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<DataExprNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
