#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json lovense_out_spec() {
  return pending_operator_spec(
      "f8.lovense_out", "Lovense Out", "output", {data_port("position", "Position command.", number_schema())}, {},
      {state_field("enabled", "Enabled", "Enable output.", boolean_schema(true)),
       state_field("commandUrl", "Command URL", "Command URL.", string_schema("")),
       state_field("platformName", "Platform Name", "Platform name.", string_schema("Feel8")),
       state_field("requestTimeoutMs", "Request Timeout (ms)", "Request timeout.", integer_schema(2000, 1)),
       state_field("verifyTls", "Verify TLS", "Verify TLS.", boolean_schema(true)),
       state_field("minSendIntervalMs", "Min Send Interval (ms)", "Minimum send interval.", integer_schema(20, 0)),
       state_field("vibrate", "Vibrate", "Vibrate value.", number_schema(0.0)),
       state_field("rotate", "Rotate", "Rotate value.", number_schema(0.0)),
       state_field("pump", "Pump", "Pump value.", number_schema(0.0)),
       state_field("thrusting", "Thrusting", "Thrusting value.", number_schema(0.0)),
       state_field("fingering", "Fingering", "Fingering value.", number_schema(0.0)),
       state_field("suction", "Suction", "Suction value.", number_schema(0.0)),
       state_field("depth", "Depth", "Depth value.", number_schema(0.0)),
       state_field("oscillate", "Oscillate", "Oscillate value.", number_schema(0.0)),
       state_field("all", "All", "All functions.", number_schema(0.0)),
       state_field("strokeMin", "Stroke Min", "Stroke min.", number_schema(0.0)),
       state_field("strokeMax", "Stroke Max", "Stroke max.", number_schema(1.0)),
       state_field("stop", "Stop", "Stop command.", boolean_schema(false)),
       state_field("timeSec", "Time Sec", "Command time.", number_schema(0.0)),
       state_field("loopRunningSec", "Loop Running Sec", "Loop running seconds.", number_schema(0.0)),
       state_field("loopPauseSec", "Loop Pause Sec", "Loop pause seconds.", number_schema(0.0)),
       state_field("stopPrevious", "Stop Previous", "Stop previous.", boolean_schema(false)),
       state_field("toy", "Toy", "Toy id.", string_schema("")),
       state_field("defaultToy", "Default Toy", "Default toy.", string_schema("")),
       state_field("availableToys", "Available Toys", "Available toys.", array_schema(string_schema("")), "ro")},
      {"sendPositionCmd", "sendFunctionCmd"});
}

}  // namespace

void register_lovense_out_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, lovense_out_spec());
}

}  // namespace f8::cppengine
