#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json program_wave_spec() {
  return pending_operator_spec(
      "f8.program_wave", "Program Wave", "wave", {}, {data_port("phaseTurns", "Phase turns.", number_schema()),
                                                        data_port("phase", "Phase.", number_schema()),
                                                        data_port("active", "Active.", boolean_schema(false)),
                                                        data_port("done", "Done.", boolean_schema(false)),
                                                        data_port("elapsedSec", "Elapsed seconds.", number_schema())},
      {state_field("program", "Program", "Program definition.", any_schema(), "rw", true, true)});
}

}  // namespace

void register_program_wave_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, program_wave_spec());
}

}  // namespace f8::cppengine
