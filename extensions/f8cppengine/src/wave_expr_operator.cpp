#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json wave_expr_spec() {
  return pending_operator_spec(
      "f8.wave_expr", "Wave Expr", "wave", {data_port("t", "Time/phase input.", number_schema())},
      {data_port("value", "Value.", number_schema())},
      {state_field("template", "Template", "Expression template.", string_schema("sin"), "rw", true, true),
       state_field("maxT", "Max T", "Maximum t.", number_schema(1.0, 0.0), "rw", true, true),
       state_field("minValue", "Min Value", "Minimum output.", number_schema(0.0)),
       state_field("maxValue", "Max Value", "Maximum output.", number_schema(1.0)),
       state_field("express", "Express", "Expression text.", string_schema(""), "rw", true, true),
       state_field("preview", "Preview", "Preview points.", any_schema(), "ro", true, false)},
      {}, {}, json{{"stateFields", editable_collection_policy()}});
}

}  // namespace

void register_wave_expr_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, wave_expr_spec());
}

}  // namespace f8::cppengine
