#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json bone_selector_spec() {
  return pending_operator_spec(
      "f8.bone_selector", "Bone Selector", "motion", {data_port("skeleton", "Skeleton input.", any_schema())},
      {data_port("bone", "Selected bone.", any_schema())},
      {state_field("target", "Target", "Target bone name.", string_schema(""), "rw", true, true),
       state_field("availableBones", "Available Bones", "Available bone names.", array_schema(string_schema("")), "ro", true, false)});
}

}  // namespace

void register_bone_selector_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, bone_selector_spec());
}

}  // namespace f8::cppengine
