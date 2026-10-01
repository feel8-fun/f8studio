#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json state_trigger_spec() {
  return pending_operator_spec(
      "f8.state_trigger", "State Trigger", "state", {}, {},
      {state_field("value", "Value", "Watched state value.", any_schema(), "rw", true, true),
       state_field("enabled", "Enabled", "Enable trigger.", boolean_schema(true), "rw", true, true),
       state_field("fireOnStart", "Fire On Start", "Fire on activation.", boolean_schema(false))},
      {}, {"changed"});
}

}  // namespace

void register_state_trigger_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, state_trigger_spec());
}

}  // namespace f8::cppengine
