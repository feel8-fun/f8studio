#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json skeleton_decoder_spec() {
  return pending_operator_spec(
      "f8.skeleton_decoder", "Skeleton Decoder", "motion", {data_port("packet", "Skeleton packet.", any_schema())},
      {data_port("skeletons", "Skeletons.", any_schema()), data_port("selectedSkeleton", "Selected skeleton.", any_schema())},
      {state_field("cleanupAfterMs", "Cleanup After (ms)", "Cleanup timeout.", integer_schema(1000, 0)),
       state_field("selectedKey", "Selected Key", "Selected skeleton key.", string_schema(""), "rw", true, true),
       state_field("availableKeys", "Available Keys", "Available skeleton keys.", array_schema(string_schema("")), "ro", true, false)},
      {"packet"}, {"packet"});
}

}  // namespace

void register_skeleton_decoder_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, skeleton_decoder_spec());
}

}  // namespace f8::cppengine
