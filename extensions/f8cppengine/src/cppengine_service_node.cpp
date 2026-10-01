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
class CppEngineServiceNode final : public f8::cppsdk::ServiceNode {
 public:
  CppEngineServiceNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : ServiceNode(node_id, data_port_names(node.dataInPorts, {}), data_port_names(node.dataOutPorts, {}),
                    state_names(node.stateFields, {"dataDelivery"})) {
    data_delivery_ = initial_state.value("dataDelivery", "buffered");
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "dataDelivery") {
      const std::string mode = value.is_string() ? value.get<std::string>() : "";
      if (mode != "buffered" && mode != "callback") {
        throw std::invalid_argument("dataDelivery must be 'buffered' or 'callback'");
      }
      return mode;
    }
    return value;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "dataDelivery" && value.is_string()) {
      data_delivery_ = value.get<std::string>();
    }
  }

 private:
  std::string data_delivery_ = "buffered";
};

json service_spec() {
  return json{{"specKind", "service"},
              {"schemaVersion", "f8service/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", "svc"},
              {"version", "0.0.1"},
              {"label", "CppEngine"},
              {"description", "C++ execution engine for high-frequency Feel8 operator graphs."},
              {"tags", json::array({"engine", "cpp", "native"})},
              {"rendererClass", "default_container"},
              {"stateFields",
               json::array({state_field("dataDelivery", "Data Delivery",
                                         "How data inputs are delivered to local nodes.", string_enum_schema("buffered", {"buffered", "callback"}),
                                         "rw", true, true)})}};
}

}  // namespace

void register_cppengine_service_node(RuntimeNodeRegistry& registry) {
  registry.register_service_spec(service_spec(), true);
  registry.register_service_factory(kServiceClass,
                                    [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                      return std::make_unique<CppEngineServiceNode>(node_id, node, initial_state);
                                    },
                                    true);
}

}  // namespace f8::cppengine
