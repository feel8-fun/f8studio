#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace f8::cppsdk {

struct DataRoute {
  std::string to_node_id;
  std::string to_port;
  std::string from_service_id;
  std::string from_node_id;
  std::string from_port;
};

// Parse a rungraph JSON object and extract cross-service data routes targeting `to_service_id`.
//
// Expected edge schema: matches `schemas/protocol.yml#/components/schemas/F8Edge`.
// Returns: map(subject -> routes), where subject is `svc.<fromServiceId>.nodes.<fromNodeId>.data.<fromPort>`.
std::unordered_map<std::string, std::vector<DataRoute>> parse_cross_service_data_routes(
    const nlohmann::json& graph_obj, const std::string& to_service_id);

}  // namespace f8::cppsdk

