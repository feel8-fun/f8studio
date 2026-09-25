#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json state_expr_spec() {
  return pending_operator_spec(
      "f8.state_expr", "State Expr", "state", {}, {},
      {state_field("allowNumpy", "Allow Numpy", "Python-only compatibility flag; ignored by C++.", boolean_schema(false)),
       state_field("code", "Code", "Expression code.", string_schema("out = 0"), "rw", true, true,
                   json{{"kind", "textarea"}, {"language", "cpp"}}),
       state_field("out", "Out", "Expression output.", any_schema(), "rw", true, true)},
      {}, {}, json{{"stateFields", editable_collection_policy()}});
}

}  // namespace

void register_state_expr_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, state_expr_spec());
}

}  // namespace f8::cppengine
