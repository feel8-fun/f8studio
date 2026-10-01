#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json object_schema(std::initializer_list<std::pair<const std::string, json>> properties) {
  json properties_object = json::object();
  for (const auto& [name, schema] : properties) {
    properties_object[name] = schema;
  }
  return json{{"type", "object"}, {"properties", properties_object}};
}

json exec_branch_spec() {
  return pending_operator_spec(
      "f8.exec_branch", "Exec Branch", "execution", {}, {},
      {state_field("selectedBranch", "Selected Branch", "Exec output port to emit for each trigger.",
                   string_schema("branch_a"), "rw", true, true),
       state_field("resolvedBranch", "Resolved Branch", "Readonly branch output actually emitted after fallback.",
                   string_schema(""), "ro", true, true)},
      {"exec"}, {"branch_a", "branch_b", "branch_c", "default"},
      json{{"execOutPorts", editable_collection_policy()}});
}

json exec_merge_spec() {
  return pending_operator_spec("f8.exec_merge", "Exec Merge", "execution", {}, {}, {},
                               {"branch_a", "branch_b", "branch_c"}, {"exec"},
                               json{{"execInPorts", editable_collection_policy()}});
}

json data_mux_spec() {
  return pending_operator_spec(
      "f8.data_mux", "Data Mux", "flow",
      {data_port("branch_a", "Branch A input.", any_schema()),
       data_port("branch_b", "Branch B input.", any_schema()),
       data_port("branch_c", "Branch C input.", any_schema()),
       data_port("default", "Fallback input.", any_schema())},
      {data_port("out", "Selected data output.", any_schema())},
      {state_field("selectedInput", "Selected Input", "Data input port to pull for the selected output.",
                   string_schema("branch_a"), "rw", true, true),
       state_field("resolvedInput", "Resolved Input", "Readonly input port actually pulled after fallback.",
                   string_schema(""), "ro", true, true)},
      {"exec"}, {"exec"}, json{{"dataInPorts", editable_collection_policy()}});
}

json skeleton_selector_spec() {
  const json status_schema = object_schema({
      {"valid", boolean_schema(false)},
      {"stableKey", string_schema("")},
      {"profileId", string_schema("")},
      {"role", string_schema("")},
      {"roleIndex", integer_schema(0)},
      {"reason", string_schema("")},
  });
  return pending_operator_spec(
      "f8.skeleton_selector", "Skeleton Selector", "motion",
      {data_port("skeletons", "Decoded skeleton list.", any_schema(), true)},
      {data_port("skeleton", "Selected skeleton.", any_schema(), true),
       data_port("stableKey", "Stable profile/role/index key.", string_schema(""), true),
       data_port("status", "Selection status on the data channel.", status_schema, true)},
      {state_field("profileId", "Profile ID", "Exporter game profile ID. Empty accepts any profile.",
                   string_schema(""), "rw", true, true),
       state_field("role", "Role", "Stable character role.",
                   string_enum_schema("", {"", "male", "female", "other"}), "rw", true, true),
       state_field("roleIndex", "Role Index", "Zero-based index within the selected role.",
                   integer_schema(0, 0, 1024), "rw", true, true),
       state_field("fallbackModelName", "Legacy Model", "Exact modelName used only for LMEX v1 streams.",
                   string_schema(""), "rw", true, false),
       state_field("allowLegacyFallback", "Legacy Fallback", "Allow exact modelName fallback for LMEX v1 packets.",
                   boolean_schema(true), "rw", true, false),
       state_field("availableKeys", "Available Characters", "Low-frequency list of currently available stable keys.",
                   array_schema(string_schema("")), "ro", true, false)});
}

json relative_pose_axes_spec() {
  const json vector_schema = array_schema(number_schema());
  const json bone_schema = object_schema({{"pos", vector_schema}, {"rot", vector_schema}});
  const json status_schema = object_schema({
      {"valid", boolean_schema(false)},       {"reason", string_schema("")},
      {"primaryAxis", string_schema("")},   {"L0", number_schema()},
      {"L1", number_schema()},               {"L2", number_schema()},
      {"R0", number_schema()},               {"R1", number_schema()},
      {"R2", number_schema()},
  });
  return pending_operator_spec(
      "f8.relative_pose_axes", "Relative Pose Axes", "motion",
      {data_port("referenceBone", "Reference bone with pos and rot.", bone_schema, true),
       data_port("targetBone", "Target bone with pos and rot.", bone_schema, true)},
      {data_port("L0", "Raw relative L0 signal.", number_schema(), true),
       data_port("L1", "Raw relative L1 signal.", number_schema(), true),
       data_port("L2", "Raw relative L2 signal.", number_schema(), true),
       data_port("R0", "Raw relative R0 signal.", number_schema(), true),
       data_port("R1", "Raw relative R1 signal.", number_schema(), true),
       data_port("R2", "Raw relative R2 signal.", number_schema(), true),
       data_port("status", "Per-sample pose calculation status.", status_schema, true)},
      {state_field("primaryAxis", "Primary Axis", "Reference-local axis used for L0.",
                   string_enum_schema("local_y", {"local_x", "local_y", "local_z", "distance"}), "rw", true,
                   true),
       state_field("invertPrimary", "Invert L0", "Invert the raw L0 direction before normalization.",
                   boolean_schema(false), "rw", true, true)});
}

json stream_watchdog_spec() {
  const json status_schema = object_schema({
      {"valid", boolean_schema(false)},
      {"ageMs", number_schema(0.0, 0.0)},
      {"timeoutMs", integer_schema(250, 10)},
      {"reason", string_schema("")},
  });
  return pending_operator_spec(
      "f8.stream_watchdog", "Stream Watchdog", "motion",
      {data_port("value", "Timestamped stream value.", any_schema(), true)},
      {data_port("value", "Input while fresh, otherwise null.", any_schema(), true),
       data_port("valid", "Whether the input is fresh.", boolean_schema(false), true),
       data_port("ageMs", "Age of the oldest input sample.", number_schema(0.0, 0.0), true),
       data_port("status", "Per-check freshness status.", status_schema, true)},
      {state_field("timeoutMs", "Timeout (ms)", "Maximum input age before output and exec flow are blocked.",
                   integer_schema(250, 10, 60000), "rw", true, true)},
      {"check"}, {"valid"});
}

}  // namespace

void register_flow_motion_pending_operators(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, exec_branch_spec());
  register_pending_operator_spec(registry, exec_merge_spec());
  register_pending_operator_spec(registry, data_mux_spec());
  register_pending_operator_spec(registry, skeleton_selector_spec());
  register_pending_operator_spec(registry, relative_pose_axes_spec());
  register_pending_operator_spec(registry, stream_watchdog_spec());
}

}  // namespace f8::cppengine
