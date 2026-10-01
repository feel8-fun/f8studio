#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json replayer_spec() {
  return pending_operator_spec(
      "f8.replayer", "Replayer", "io", {}, {data_port("positionMs", "Replay position in ms.", number_schema())},
      {state_field("path", "Path", "Replay path.", string_schema(""), "rw", true, true),
       state_field("loop", "Loop", "Loop replay.", boolean_schema(false)),
       state_field("timeMode", "Time Mode", "Replay time mode.", string_schema("wall")),
       state_field("playing", "Playing", "Playing.", boolean_schema(false), "ro", true, true),
       state_field("durationMs", "Duration (ms)", "Duration.", integer_schema(0, 0), "ro"),
       state_field("loaded", "Loaded", "Loaded.", boolean_schema(false), "ro")},
      {"play", "pause", "stop"}, {"sample", "started", "stopped", "looped", "done"});
}

}  // namespace

void register_replayer_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, replayer_spec());
}

}  // namespace f8::cppengine
