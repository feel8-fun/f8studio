#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json recorder_spec() {
  return pending_operator_spec(
      "f8.recorder", "Recorder", "io", {}, {},
      {state_field("path", "Path", "Recording path.", string_schema(""), "rw", true, true),
       state_field("enabled", "Enabled", "Enable recording.", boolean_schema(true), "rw", true, true),
       state_field("append", "Append", "Append to existing file.", boolean_schema(false)),
       state_field("recording", "Recording", "Recording state.", boolean_schema(false), "ro", true, true),
       state_field("sessionStartTsMs", "Session Start", "Session start timestamp.", integer_schema(0, 0), "ro")},
      {"record"});
}

}  // namespace

void register_recorder_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, recorder_spec());
}

}  // namespace f8::cppengine
