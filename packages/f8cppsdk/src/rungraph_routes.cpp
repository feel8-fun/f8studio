#include "f8cppsdk/rungraph_routes.h"

#include <utility>

#include <nlohmann/json.hpp>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/generated/protocol_models.h"

namespace f8::cppsdk {

using json = nlohmann::json;

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

    std::string subject;
    try {
      subject = data_subject(from_sid, from_nid, from_port);
    } catch (...) {
      continue;
    }

    DataRoute r;
    r.to_node_id = std::move(to_nid);
    r.to_port = to_port;
    r.from_service_id = from_sid;
    r.from_node_id = from_nid;
    r.from_port = from_port;
    routes[subject].push_back(std::move(r));
  }

  return routes;
}

}  // namespace f8::cppsdk
