#include "f8cppsdk/rungraph_routes.h"

#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/generated/protocol_models.h"

namespace f8::cppsdk {

using json = nlohmann::json;

namespace {

bool is_stream_payload_kind(const std::string& kind) {
  return kind == "bytes" || kind == "video_frame" || kind == "audio_chunk";
}

const json* find_runtime_node(const json& graph_obj, const std::string& node_id) {
  if (node_id.empty()) {
    return nullptr;
  }
  const auto nodes_it = graph_obj.find("nodes");
  if (nodes_it == graph_obj.end() || !nodes_it->is_array()) {
    return nullptr;
  }
  for (const auto& node : *nodes_it) {
    if (!node.is_object()) {
      continue;
    }
    const auto node_id_it = node.find("nodeId");
    if (node_id_it != node.end() && node_id_it->is_string() && node_id_it->get<std::string>() == node_id) {
      return &node;
    }
  }
  return nullptr;
}

std::string port_payload_kind(const json* node, const std::string& port_name, bool output_port) {
  if (node == nullptr || port_name.empty()) {
    return "json";
  }
  const char* field = output_port ? "dataOutPorts" : "dataInPorts";
  const auto ports_it = node->find(field);
  if (ports_it == node->end() || !ports_it->is_array()) {
    return "json";
  }
  for (const auto& port : *ports_it) {
    if (!port.is_object()) {
      continue;
    }
    const auto name_it = port.find("name");
    if (name_it == port.end() || !name_it->is_string() || name_it->get<std::string>() != port_name) {
      continue;
    }
    const auto payload_it = port.find("payload");
    if (payload_it != port.end() && payload_it->is_object()) {
      const auto kind_it = payload_it->find("kind");
      if (kind_it != payload_it->end() && kind_it->is_string()) {
        return kind_it->get<std::string>();
      }
    }
    throw std::invalid_argument("data port requires an explicit payload.kind");
  }
  return "json";
}

bool edge_uses_stream_payload(const json& graph_obj, const generated::F8Edge& edge) {
  std::string from_node_id = edge.fromOperatorId.value_or("");
  if (from_node_id.empty()) {
    from_node_id = edge.fromServiceId;
  }
  std::string to_node_id = edge.toOperatorId.value_or("");
  if (to_node_id.empty()) {
    to_node_id = edge.toServiceId;
  }
  const json* from_node = find_runtime_node(graph_obj, from_node_id);
  const json* to_node = find_runtime_node(graph_obj, to_node_id);
  if (is_stream_payload_kind(port_payload_kind(from_node, edge.fromPort, true))) {
    return true;
  }
  if (is_stream_payload_kind(port_payload_kind(to_node, edge.toPort, false))) {
    return true;
  }
  return false;
}

}  // namespace

std::unordered_map<std::string, std::vector<DataRoute>> parse_cross_service_data_routes(
    const json& graph_obj, const std::string& to_service_id) {
  std::unordered_map<std::string, std::vector<DataRoute>> routes;

  if (!graph_obj.is_object()) {
    return routes;
  }
  const auto edges_it = graph_obj.find("edges");
  if (edges_it == graph_obj.end() || !edges_it->is_array()) {
    return routes;
  }

  const std::string to_sid = to_service_id;
  if (to_sid.empty()) {
    return routes;
  }

  for (const auto& e : *edges_it) {
    generated::F8Edge edge{};
    generated::ParseError err{};
    if (!generated::parse_F8Edge(e, edge, err)) {
      continue;
    }

    if (edge.kind != generated::F8EdgeKindEnum::data) continue;

    const std::string from_sid = edge.fromServiceId;
    const std::string edge_to_sid = edge.toServiceId;
    if (from_sid.empty() || edge_to_sid.empty()) continue;
    if (edge_to_sid != to_sid) continue;

    // Cross-service only.
    if (from_sid == edge_to_sid) continue;

    std::string from_nid = edge.fromOperatorId.value_or("");
    if (from_nid.empty()) from_nid = from_sid;  // service node
    std::string to_nid = edge.toOperatorId.value_or("");
    if (to_nid.empty()) to_nid = edge_to_sid;  // service node

    const std::string from_port = edge.fromPort;
    const std::string to_port = edge.toPort;
    if (from_port.empty() || to_port.empty()) continue;

    std::string key;
    try {
      key = data_key(from_sid, from_nid, from_port);
    } catch (...) {
      continue;
    }

    DataRoute r;
    r.to_node_id = std::move(to_nid);
    r.to_port = to_port;
    r.from_service_id = from_sid;
    r.from_node_id = from_nid;
    r.from_port = from_port;
    r.stream_payload = edge_uses_stream_payload(graph_obj, edge);
    if (edge.strategy.has_value() && edge.strategy.value() == generated::F8EdgeStrategyEnum::queue) {
      r.strategy = EdgeStrategy::kQueue;
    } else {
      r.strategy = EdgeStrategy::kLatest;
    }
    if (edge.timeoutMs.has_value()) {
      r.timeout_ms = edge.timeoutMs.value();
    }
    routes[key].push_back(std::move(r));
  }

  return routes;
}

}  // namespace f8::cppsdk
