#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json sequence_player_spec() {
  return pending_operator_spec(
      "f8.sequence_player", "Sequence Player", "wave", {}, {data_port("value", "Value.", any_schema()),
                                                             data_port("index", "Index.", integer_schema(0)),
                                                             data_port("active", "Active.", boolean_schema(false)),
                                                             data_port("done", "Done.", boolean_schema(false)),
                                                             data_port("elapsedSec", "Elapsed seconds.", number_schema())},
      {state_field("sequence", "Sequence", "Sequence definition.", any_schema(), "rw", true, true)});
}

}  // namespace

void register_sequence_player_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, sequence_player_spec());
}

}  // namespace f8::cppengine
