#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json playback_sync_spec() {
  return pending_operator_spec(
      "f8.playback_sync", "Playback Sync", "playback", {data_port("playback", "Playback snapshot.", any_schema())},
      {data_port("position", "Position.", number_schema()), data_port("rawPosition", "Raw position.", number_schema()),
       data_port("duration", "Duration.", number_schema()), data_port("playing", "Playing.", boolean_schema(false)),
       data_port("videoId", "Video id.", string_schema("")), data_port("ageMs", "Age in ms.", number_schema()),
       data_port("stale", "Stale.", boolean_schema(false))},
      {state_field("maxExtrapolateMs", "Max Extrapolate (ms)", "Maximum extrapolation time.", integer_schema(500, 0)),
       state_field("playbackRate", "Playback Rate", "Playback rate.", number_schema(1.0, 0.0)),
       state_field("clampToDuration", "Clamp To Duration", "Clamp position to duration.", boolean_schema(true))});
}

}  // namespace

void register_playback_sync_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, playback_sync_spec());
}

}  // namespace f8::cppengine
