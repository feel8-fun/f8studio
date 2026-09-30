#include "f8cppsdk/service_host.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "f8cppsdk/time_utils.h"

namespace f8::cppsdk {

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(const SteadyClock::time_point start, const SteadyClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

ServiceHost::ServiceHost(ServiceBus& bus, RuntimeNodeRegistry& registry, std::string service_class)
    : bus_(bus), registry_(registry), service_class_(std::move(service_class)) {}

ServiceHost::~ServiceHost() { stop(); }

void ServiceHost::start() {
  if (started_) return;
  ensure_service_node();
  bus_.add_lifecycle_node(this);
  bus_.add_stateful_node(this);
  bus_.add_data_node(this);
  bus_.add_set_state_node(this);
  started_ = true;
}

void ServiceHost::stop() {
  for (auto& item : operator_nodes_) {
    if (item.second) close_node(*item.second);
  }
  operator_nodes_.clear();
  if (service_node_) {
    close_node(*service_node_);
    service_node_.reset();
  }
  started_ = false;
}

bool ServiceHost::apply_rungraph(const nlohmann::json& graph_obj, std::string& error_code,
                                 std::string& error_message) {
  error_code.clear();
  error_message.clear();

  generated::F8RuntimeGraph graph;
  generated::ParseError parse_error;
  if (!generated::parse_F8RuntimeGraph(graph_obj, graph, parse_error)) {
    error_code = "INVALID_RUNGRAPH";
    error_message = parse_error.message.empty() ? "invalid rungraph" : parse_error.message;
    return false;
  }

  ensure_service_node();
  const std::string service_id = bus_.config().service_id;

  generated::F8RuntimeNode service_snapshot;
  bool has_service_snapshot = false;
  std::vector<generated::F8RuntimeNode> wanted;
  for (const auto& node : graph.nodes.value_or(std::vector<generated::F8RuntimeNode>{})) {
    if (node.serviceId != service_id || node.serviceClass != service_class_) {
      continue;
    }
    const bool is_service_node = !node.operatorClass.has_value() || node.operatorClass.value().empty();
    if (is_service_node) {
      if (node.nodeId == service_id) {
        service_snapshot = node;
        has_service_snapshot = true;
      }
      continue;
    }
    wanted.push_back(node);
  }

  if (has_service_snapshot && service_node_) {
    service_node_->set_data_in_ports(data_port_names(service_snapshot.dataInPorts));
    service_node_->set_data_out_ports(data_port_names(service_snapshot.dataOutPorts));
    service_node_->set_state_fields(state_field_names(service_snapshot.stateFields));
  }

  std::unordered_set<std::string> wanted_ids;
  for (const auto& node : wanted) {
    wanted_ids.insert(node.nodeId);
  }
  for (auto it = operator_nodes_.begin(); it != operator_nodes_.end();) {
    if (wanted_ids.find(it->first) != wanted_ids.end()) {
      ++it;
      continue;
    }
    if (it->second) close_node(*it->second);
    it = operator_nodes_.erase(it);
  }

  for (const auto& node : wanted) {
    const std::string node_id = node.nodeId;
    auto existing_it = operator_nodes_.find(node_id);
    if (existing_it != operator_nodes_.end()) {
      if (existing_it->second && needs_recreate(*existing_it->second, node)) {
        close_node(*existing_it->second);
        operator_nodes_.erase(existing_it);
      } else {
        continue;
      }
    }

    try {
      auto runtime_node = registry_.create_operator_node(node_id, node, initial_state(node));
      runtime_node->set_data_in_ports(data_port_names(node.dataInPorts));
      runtime_node->set_data_out_ports(data_port_names(node.dataOutPorts));
      runtime_node->set_state_fields(state_field_names(node.stateFields));
      runtime_node->set_exec_in_ports(string_vector(node.execInPorts));
      runtime_node->set_exec_out_ports(string_vector(node.execOutPorts));
      runtime_node->attach(&bus_);
      operator_nodes_[node_id] = std::move(runtime_node);
    } catch (const OperatorFactoryNotRegistered& exc) {
      spdlog::error("missing C++ operator runtime factory nodeId={} operatorClass={}: {}", node_id,
                    node.operatorClass.value_or(""), exc.what());
    } catch (const std::exception& exc) {
      spdlog::error("failed to create C++ runtime node nodeId={}: {}", node_id, exc.what());
    }
  }

  return true;
}

RuntimeNode* ServiceHost::get_node(const std::string& node_id) const {
  if (service_node_ && service_node_->node_id() == node_id) {
    return service_node_.get();
  }
  const auto it = operator_nodes_.find(node_id);
  if (it == operator_nodes_.end()) return nullptr;
  return it->second.get();
}

std::vector<OperatorNode*> ServiceHost::operator_nodes() const {
  std::vector<OperatorNode*> out;
  out.reserve(operator_nodes_.size());
  for (const auto& item : operator_nodes_) {
    if (item.second) out.push_back(item.second.get());
  }
  std::sort(out.begin(), out.end(), [](const OperatorNode* a, const OperatorNode* b) {
    if (a == nullptr || b == nullptr) return a < b;
    return a->node_id() < b->node_id();
  });
  return out;
}

void ServiceHost::on_lifecycle(bool active, const nlohmann::json& meta) {
  if (service_node_) {
    service_node_->on_lifecycle(active, meta);
  }
  for (auto& item : operator_nodes_) {
    if (item.second) item.second->on_lifecycle(active, meta);
  }
}

void ServiceHost::on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                           std::int64_t ts_ms, const nlohmann::json& meta) {
  RuntimeNode* node = get_node(node_id);
  if (node == nullptr) return;
  try {
    node->on_state(field, value, ts_ms, meta);
  } catch (const std::exception& exc) {
    bus_.report_error(node_id, "STATE_CALLBACK_FAILED", exc.what());
  } catch (...) {
    bus_.report_error(node_id, "STATE_CALLBACK_FAILED", "on_state threw unknown exception");
  }
}

void ServiceHost::on_data(const std::string& node_id, const std::string& port, const nlohmann::json& value,
                          std::int64_t ts_ms, const nlohmann::json& meta) {
  RuntimeNode* node = get_node(node_id);
  if (node == nullptr) return;
  const auto start = SteadyClock::now();
  try {
    node->on_data(port, value, ts_ms, meta);
    const auto end = SteadyClock::now();
    const double latency_ms = elapsed_ms(start, end);
    bus_.record_monitor_timing("data", latency_ms, latency_ms, now_ms());
  } catch (const std::exception& exc) {
    const auto end = SteadyClock::now();
    const double latency_ms = elapsed_ms(start, end);
    bus_.record_monitor_timing("data", latency_ms, latency_ms, now_ms());
    bus_.report_error(node_id, "DATA_CALLBACK_FAILED", exc.what(), "error", "", ts_ms);
  } catch (...) {
    const auto end = SteadyClock::now();
    const double latency_ms = elapsed_ms(start, end);
    bus_.record_monitor_timing("data", latency_ms, latency_ms, now_ms());
    bus_.report_error(node_id, "DATA_CALLBACK_FAILED", "on_data threw unknown exception", "error", "", ts_ms);
  }
}

bool ServiceHost::on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                               const nlohmann::json& meta, std::string& error_code, std::string& error_message) {
  RuntimeNode* node = get_node(node_id);
  if (node == nullptr) {
    error_code = "UNKNOWN_NODE";
    error_message = "unknown runtime node";
    return false;
  }
  try {
    const nlohmann::json normalized = node->validate_state(field, value, 0, meta);
    if (!bus_.publish_state_from_external(node_id, field, normalized, meta)) {
      error_code = "INTERNAL";
      error_message = "failed to publish state";
      return false;
    }
    error_code.clear();
    error_message.clear();
    return true;
  } catch (const std::exception& exc) {
    error_code = "INVALID_VALUE";
    error_message = exc.what();
    return false;
  }
}

std::vector<std::string> ServiceHost::data_port_names(
    const std::optional<std::vector<generated::F8DataPortSpec>>& ports) {
  std::vector<std::string> out;
  for (const auto& port : ports.value_or(std::vector<generated::F8DataPortSpec>{})) {
    out.push_back(port.name);
  }
  return out;
}

std::vector<std::string> ServiceHost::state_field_names(
    const std::optional<std::vector<generated::F8StateSpec>>& fields) {
  std::vector<std::string> out;
  for (const auto& field : fields.value_or(std::vector<generated::F8StateSpec>{})) {
    out.push_back(field.name);
  }
  return out;
}

std::vector<std::string> ServiceHost::string_vector(const std::optional<std::vector<std::string>>& values) {
  return values.value_or(std::vector<std::string>{});
}

nlohmann::json ServiceHost::initial_state(const generated::F8RuntimeNode& node) {
  return node.stateValues ? generated::wire_json(*node.stateValues) : nlohmann::json::object();
}

bool ServiceHost::needs_recreate(const OperatorNode& node, const generated::F8RuntimeNode& snapshot) {
  return node.data_in_ports() != data_port_names(snapshot.dataInPorts) ||
         node.data_out_ports() != data_port_names(snapshot.dataOutPorts) ||
         node.state_fields() != state_field_names(snapshot.stateFields) ||
         node.exec_in_ports() != string_vector(snapshot.execInPorts) ||
         node.exec_out_ports() != string_vector(snapshot.execOutPorts);
}

void ServiceHost::ensure_service_node() {
  if (service_node_) return;
  generated::F8RuntimeNode node;
  node.nodeId = bus_.config().service_id;
  node.serviceId = bus_.config().service_id;
  node.serviceClass = service_class_;
  service_node_ = registry_.create_service_node(service_class_, bus_.config().service_id, node, nlohmann::json::object());
  service_node_->attach(&bus_);
}

void ServiceHost::close_node(RuntimeNode& node) {
  try {
    if (auto* closeable = dynamic_cast<ClosableNode*>(&node)) {
      closeable->close();
    }
  } catch (const std::exception& exc) {
    spdlog::warn("failed to close runtime node nodeId={}: {}", node.node_id(), exc.what());
  }
  node.detach();
}

}  // namespace f8::cppsdk
