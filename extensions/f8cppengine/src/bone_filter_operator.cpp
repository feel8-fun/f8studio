#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json bone_filter_spec() {
  return pending_operator_spec(
      "f8.bone_filter", "Bone Filter", "motion", {data_port("bone", "Bone input.", any_schema())},
      {data_port("filtered", "Filtered bone.", any_schema()), data_port("relative", "Relative bone.", any_schema())},
      {state_field("filter_type", "Filter", "Filter type.", string_schema("EMA"), "rw", true, true),
       state_field("ema_alpha", "EMA Alpha", "EMA smoothing factor.", number_schema(0.4, 0.0, 1.0), "rw", true, true),
       state_field("dema_alpha", "DEMA Alpha", "DEMA smoothing factor.", number_schema(0.4, 0.0, 1.0)),
       state_field("one_euro_min_cutoff", "One Euro Min Cutoff", "Minimum cutoff.", number_schema(1.5, 0.01, 10.0)),
       state_field("one_euro_beta", "One Euro Beta", "Speed coefficient.", number_schema(0.0, 0.0, 5.0)),
       state_field("one_euro_derivative_cutoff", "One Euro Derivative Cutoff", "Derivative cutoff.", number_schema(1.0, 0.01, 10.0)),
       state_field("one_euro_default_freq", "One Euro Default Freq", "Default frequency.", number_schema(90.0, 1.0, 240.0)),
       state_field("jumpEnabled", "Jump Enabled", "Enable jump rejection.", boolean_schema(true)),
       state_field("jumpPosThreshold", "Jump Pos Threshold", "Position jump threshold.", number_schema(1.0, 0.0)),
       state_field("jumpRotDegThreshold", "Jump Rot Deg Threshold", "Rotation jump threshold.", number_schema(45.0, 0.0)),
       state_field("jumpConsecutiveFrames", "Jump Consecutive Frames", "Consecutive frames.", integer_schema(2, 1)),
       state_field("jumpCooldownFrames", "Jump Cooldown Frames", "Cooldown frames.", integer_schema(10, 0))});
}

}  // namespace

void register_bone_filter_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, bone_filter_spec());
}

}  // namespace f8::cppengine
