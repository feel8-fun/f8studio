#include "f8cppsdk/describe_builtins.h"

#include <cstdint>
#include <string>
#include <unordered_set>

#include "f8cppsdk/describe_schema.h"

namespace f8::cppsdk {

using json = nlohmann::json;

namespace {

json schema_boolean_with_default(const bool value) {
  return json{{"type", "boolean"}, {"default", value}};
}

json schema_number(const double default_value) {
  return json{{"type", "number"}, {"default", default_value}, {"minimum", 0.0}};
}

json schema_integer(const std::int64_t default_value) {
  return json{{"type", "integer"}, {"default", default_value}, {"minimum", 0}};
}

json locked_builtin_state_policy() {
  return json{{"canRename", false},
              {"canEditAccess", false},
              {"canEditValueRequired", false},
              {"canEditValueSchema", false}};
}

json state_field(const std::string& name, const json& value_schema, const std::string& access, const std::string& label,
                 const std::string& description, const bool required, const bool show_on_node) {
  return json{
      {"name", name},
      {"label", label},
      {"description", description},
      {"valueSchema", value_schema},
      {"access", access},
      {"valueRequired", required},
      {"editPolicy", locked_builtin_state_policy()},
      {"showOnNode", show_on_node},
  };
}

json monitor_port_schema() {
  const json cpu = json{
      {"type", "object"},
      {"properties", json{{"processPercent", schema_number(0.0)}, {"systemPercent", schema_number(0.0)}}},
  };
  const json memory = json{
      {"type", "object"},
      {"properties", json{{"rssBytes", schema_integer(0)}, {"vmsBytes", schema_integer(0)}}},
  };
  const json gpu = json{
      {"type", "object"},
      {"properties", json{{"vendor", json{{"type", "string"}, {"default", ""}}},
                           {"deviceIndex", json{{"type", "integer"}, {"default", -1}}},
                           {"utilPercent", schema_number(0.0)},
                           {"memoryUsedBytes", schema_integer(0)},
                           {"memoryTotalBytes", schema_integer(0)},
                           {"available", json{{"type", "boolean"}, {"default", false}}}}},
  };
  const json frame = json{
      {"type", "object"},
      {"properties", json{{"observed", schema_integer(0)},
                           {"processed", schema_integer(0)},
                           {"dropped", schema_integer(0)},
                           {"localOnlyEmits", schema_integer(0)},
                           {"routedCrossEmits", schema_integer(0)},
                           {"suppressedCrossPublishes", schema_integer(0)},
                           {"callbackDeliveries", schema_integer(0)},
                           {"bufferPullDeliveries", schema_integer(0)}}},
  };
  const json timing = json{
      {"type", "object"},
      {"properties", json{{"processMsAvg", schema_number(0.0)},
                           {"processMsP95", schema_number(0.0)},
                           {"waitMsAvg", schema_number(0.0)},
                           {"waitMsP95", schema_number(0.0)},
                           {"latencyMsAvg", schema_number(0.0)},
                           {"latencyMsP95", schema_number(0.0)}}},
  };
  const json queue = json{
      {"type", "object"},
      {"properties", json{{"depth", schema_integer(0)}}},
  };
  const json error = json{
      {"type", "object"},
      {"properties", json{{"countWindow", schema_integer(0)},
                           {"lastNodeId", json{{"type", "string"}, {"default", ""}}},
                           {"lastCode", json{{"type", "string"}, {"default", ""}}},
                           {"lastMessage", json{{"type", "string"}, {"default", ""}}},
                           {"lastSeverity",
                            json{{"type", "string"},
                                 {"default", "error"},
                                 {"enum", json::array({"critical", "error", "info", "warning"})}}},
                           {"lastFingerprint", json{{"type", "string"}, {"default", ""}}},
                           {"lastRepeatCount", schema_integer(0)},
                           {"lastTsMs", schema_integer(0)},
                           {"currentNodeId", json{{"type", "string"}, {"default", ""}}},
                           {"currentCode", json{{"type", "string"}, {"default", ""}}},
                           {"currentMessage", json{{"type", "string"}, {"default", ""}}},
                           {"currentSeverity",
                            json{{"type", "string"},
                                 {"default", ""},
                                 {"enum", json::array({"", "info", "warning", "error", "critical"})}}},
                           {"currentTsMs", schema_integer(0)}}},
  };

  return json{
      {"type", "object"},
      {"properties", json{{"schemaVersion", json{{"type", "string"}, {"default", "f8monitor/1"}, {"enum", json::array({"f8monitor/1"})}}},
                           {"serviceId", json{{"type", "string"}, {"default", ""}}},
                           {"serviceClass", json{{"type", "string"}, {"default", ""}}},
                           {"nodeId", json{{"type", "string"}, {"default", ""}}},
                           {"tsMs", schema_integer(0)},
                           {"alive", json{{"type", "boolean"}, {"default", true}}},
                           {"ready", json{{"type", "boolean"}, {"default", false}}},
                           {"active", json{{"type", "boolean"}, {"default", true}}},
                           {"uptimeMs", schema_integer(0)},
                           {"cpu", cpu},
                           {"memory", memory},
                           {"gpu", gpu},
                           {"frame", frame},
                           {"timing", timing},
                           {"queue", queue},
                           {"error", error}}},
  };
}

json monitor_port_spec() {
  return json{
      {"name", "monitor"},
      {"description", "Unified runtime monitor snapshots (health/resource/perf/error)."},
      {"valueSchema", monitor_port_schema()},
      {"definitionProtected", true},
      {"showOnNode", false},
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
                                   "Service lifecycle state (activate/deactivate).", true, false));
  }
  filtered.push_back(state_field("svcId", describe::schema_string(), "ro", "Service Id",
                                 "Readonly: current service instance id (svcId).", true, false));
  if (!is_service) {
    filtered.push_back(state_field("operatorId", describe::schema_string(), "ro", "Operator Id",
                                   "Readonly: current operator/node id (operatorId).", true, false));
  }

  spec["stateFields"] = std::move(filtered);
}

void upsert_builtin_data_out_ports(json& spec) {
  json filtered = json::array();
  if (spec.contains("dataOutPorts") && spec["dataOutPorts"].is_array()) {
    for (const auto& item : spec["dataOutPorts"]) {
      if (!item.is_object()) continue;
      const std::string name = item.value("name", std::string());
      if (name == "monitor" || name == "telemetry") continue;
      filtered.push_back(item);
    }
  }
  filtered.push_back(monitor_port_spec());
  spec["dataOutPorts"] = std::move(filtered);
}

void rename_descriptor_field(json& item, const char* old_name, const char* new_name) {
  if (!item.is_object() || !item.contains(old_name)) return;
  if (!item.contains(new_name)) item[new_name] = item[old_name];
  item.erase(old_name);
}

void normalize_control(json& field) {
  if (!field.is_object() || !field.contains("uiControl")) return;
  if (!field.contains("control") && field["uiControl"].is_string()) {
    const std::string raw = field["uiControl"].get<std::string>();
    if (!raw.empty()) {
      const auto bracket = raw.find('[');
      std::string kind = raw.substr(0, bracket);
      if (kind == "wrapline") kind = "textarea";
      json control{{"kind", kind}};
      if (bracket != std::string::npos && raw.back() == ']') {
        const std::string argument = raw.substr(bracket + 1, raw.size() - bracket - 2);
        if (kind == "select" || kind == "multiselect") control["optionsFromState"] = argument;
        if (kind == "code" || kind == "textarea") control["language"] = argument;
      }
      field["control"] = std::move(control);
    }
  }
  field.erase("uiControl");
}

void normalize_spec_descriptors(json& spec, const bool is_service) {
  if (is_service) spec.erase("launch");
  if (spec.contains("stateFields") && spec["stateFields"].is_array()) {
    for (auto& field : spec["stateFields"]) {
      rename_descriptor_field(field, "required", "valueRequired");
      normalize_control(field);
      if (field.is_object() && field.contains("editPolicy") && field["editPolicy"].is_object()) {
        rename_descriptor_field(field["editPolicy"], "canEditRequired", "canEditValueRequired");
      }
    }
  }
  for (const char* collection : {"dataInPorts", "dataOutPorts"}) {
    if (!spec.contains(collection) || !spec[collection].is_array()) continue;
    for (auto& port : spec[collection]) rename_descriptor_field(port, "required", "definitionProtected");
  }
  if (spec.contains("commands") && spec["commands"].is_array()) {
    for (auto& command : spec["commands"]) {
      rename_descriptor_field(command, "required", "definitionProtected");
      if (!command.is_object() || !command.contains("params") || !command["params"].is_array()) continue;
      for (auto& param : command["params"]) {
        rename_descriptor_field(param, "required", "valueRequired");
        normalize_control(param);
      }
    }
  }
  if (is_service) return;
  for (const char* collection : {"execInPorts", "execOutPorts"}) {
    if (!spec.contains(collection) || !spec[collection].is_array()) continue;
    for (auto& port : spec[collection]) {
      if (port.is_string()) port = json{{"name", port}};
    }
  }
}

}  // namespace

json normalize_describe_with_builtin_state_fields(const json& payload) {
  if (!payload.is_object()) return payload;

  json out = payload;
  if (out.contains("service") && out["service"].is_object()) {
    json service = out["service"];
    normalize_spec_descriptors(service, true);
    upsert_builtin_state_fields(service, true);
    upsert_builtin_data_out_ports(service);
    out["service"] = std::move(service);

    json normalized_ops = json::array();
    if (out.contains("operators") && out["operators"].is_array()) {
      for (const auto& op : out["operators"]) {
        if (!op.is_object()) continue;
        json op_spec = op;
        normalize_spec_descriptors(op_spec, false);
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

  normalize_spec_descriptors(out, true);
  upsert_builtin_state_fields(out, true);
  upsert_builtin_data_out_ports(out);
  return out;
}

}  // namespace f8::cppsdk
