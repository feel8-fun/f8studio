#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json wave_funscript_spec() {
  return pending_operator_spec(
      "f8.wave_funscript", "Wave Funscript", "wave", {data_port("t", "Time/phase input.", number_schema())},
      {data_port("value", "Value.", number_schema())},
      {state_field("funscriptPath", "Funscript Path", "Funscript path.", string_schema(""), "rw", true, true),
       state_field("allAxes", "All Axes", "All axes.", any_schema(), "ro", true, false),
       state_field("selectedAxis", "Selected Axis", "Selected axis.", string_schema("")),
       state_field("points", "Points", "Loaded points.", any_schema(), "ro", true, false),
       state_field("maxT", "Max T", "Maximum t.", number_schema(1.0, 0.0)),
       state_field("interp", "Interp", "Interpolation mode.", string_schema("linear")),
       state_field("heatmap", "Heatmap", "Heatmap preview.", any_schema(), "ro", true, false)},
      {}, {}, json{{"stateFields", editable_collection_policy()}});
}

}  // namespace

void register_wave_funscript_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, wave_funscript_spec());
}

}  // namespace f8::cppengine
