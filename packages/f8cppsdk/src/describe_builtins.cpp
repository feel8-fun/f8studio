#include "f8cppsdk/describe_builtins.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace f8::cppsdk {

using json = nlohmann::json;

namespace {

json schema_boolean_with_default(const bool value) {
  return json{{"type", "boolean"}, {"default", value}};
}

json schema_string() {
  return json{{"type", "string"}};
}

json state_field(const std::string& name, const json& value_schema, const std::string& access, const std::string& label,
                 const std::string& description, const bool show_on_node) {
  return json{
      {"name", name},
      {"label", label},
      {"description", description},
      {"valueSchema", value_schema},
      {"access", access},
      {"showOnNode", show_on_node},
  };
}

void upsert_builtin_state_fields(json& spec, const bool is_service) {
  json filtered = json::array();
  std::unordered_set<std::string> blocked{"svcId", "operatorId"};
  if (is_service) {
    blocked = {"active", "svcId"};
  }

  if (spec.contains("stateFields") && spec["stateFields"].is_array()) {
    for (const auto& item : spec["stateFields"]) {
      if (!item.is_object()) continue;
      const std::string name = item.value("name", std::string());
      if (blocked.find(name) != blocked.end()) continue;
      filtered.push_back(item);
    }
  }

  if (is_service) {
    filtered.push_back(state_field("active", schema_boolean_with_default(true), "rw", "Active",
                                   "Service lifecycle state (activate/deactivate).", true));
  }
  filtered.push_back(
      state_field("svcId", schema_string(), "ro", "Service Id", "Readonly: current service instance id (svcId).", false));
  if (!is_service) {
    filtered.push_back(state_field("operatorId", schema_string(), "ro", "Operator Id",
                                   "Readonly: current operator/node id (operatorId).", false));
  }

  spec["stateFields"] = std::move(filtered);
}

}  // namespace

json normalize_describe_with_builtin_state_fields(const json& payload) {
  if (!payload.is_object()) return payload;

  json out = payload;
  if (out.contains("service") && out["service"].is_object()) {
    json service = out["service"];
    upsert_builtin_state_fields(service, true);
    out["service"] = std::move(service);

    json normalized_ops = json::array();
    if (out.contains("operators") && out["operators"].is_array()) {
      for (const auto& op : out["operators"]) {
        if (!op.is_object()) continue;
        json op_spec = op;
        upsert_builtin_state_fields(op_spec, false);
        normalized_ops.push_back(std::move(op_spec));
      }
    }
    out["operators"] = std::move(normalized_ops);
    if (!out.contains("schemaVersion") || !out["schemaVersion"].is_string()) {
      out["schemaVersion"] = "f8describe/1";
    }
    return out;
  }

  upsert_builtin_state_fields(out, true);
  return out;
}

}  // namespace f8::cppsdk
