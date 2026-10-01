#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json buttplug_out_spec() {
  return pending_operator_spec(
      "f8.buttplug_out", "Buttplug Out", "output", {data_port("position", "Position command.", number_schema())}, {},
      {state_field("enabled", "Enabled", "Enable output.", boolean_schema(true)),
       state_field("wsUrl", "WebSocket URL", "WebSocket URL.", string_schema("ws://127.0.0.1:12345")),
       state_field("autoConnect", "Auto Connect", "Auto connect.", boolean_schema(true)),
       state_field("autoScanOnConnect", "Auto Scan On Connect", "Auto scan.", boolean_schema(true)),
       state_field("scanDurationMs", "Scan Duration (ms)", "Scan duration.", integer_schema(5000, 0)),
       state_field("reconnectIntervalMs", "Reconnect Interval (ms)", "Reconnect interval.", integer_schema(2000, 0)),
       state_field("selectedDevice", "Selected Device", "Selected device.", string_schema("")),
       state_field("rescan", "Rescan", "Rescan.", boolean_schema(false)),
       state_field("vibrateFeatureIndex", "Vibrate Feature Index", "Feature index.", integer_schema(0, 0)),
       state_field("rotateFeatureIndex", "Rotate Feature Index", "Feature index.", integer_schema(0, 0)),
       state_field("oscillateFeatureIndex", "Oscillate Feature Index", "Feature index.", integer_schema(0, 0)),
       state_field("positionFeatureIndex", "Position Feature Index", "Feature index.", integer_schema(0, 0)),
       state_field("defaultPositionDurationMs", "Default Position Duration (ms)", "Default duration.", integer_schema(100, 1)),
       state_field("vibrate", "Vibrate", "Vibrate.", number_schema(0.0)),
       state_field("rotate", "Rotate", "Rotate.", number_schema(0.0)),
       state_field("oscillate", "Oscillate", "Oscillate.", number_schema(0.0)),
       state_field("stop", "Stop", "Stop.", boolean_schema(false)),
       state_field("stopOnDeactivate", "Stop On Deactivate", "Stop on deactivate.", boolean_schema(true)),
       state_field("connected", "Connected", "Connected.", boolean_schema(false), "ro"),
       state_field("scanning", "Scanning", "Scanning.", boolean_schema(false), "ro"),
       state_field("availableDevices", "Available Devices", "Available devices.", array_schema(string_schema("")), "ro"),
       state_field("deviceInfos", "Device Infos", "Device infos.", any_schema(), "ro"),
       state_field("selectedDeviceInfo", "Selected Device Info", "Selected device info.", any_schema(), "ro")},
      {"sendPositionCmd", "sendFunctionCmd"});
}

}  // namespace

void register_buttplug_out_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, buttplug_out_spec());
}

}  // namespace f8::cppengine
