#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json handy_out_spec() {
  return pending_operator_spec(
      "f8.handy_out", "Handy Out", "output",
      {data_port("value", "Position value.", number_schema()), data_port("durationMs", "Duration override.", number_schema()),
       data_port("immediateResponse", "Immediate response override.", boolean_schema(false)),
       data_port("stopOnTarget", "Stop on target override.", boolean_schema(false))},
      {data_port("sentPosition", "Sent position.", number_schema()), data_port("httpStatus", "HTTP status.", integer_schema(0, 0)),
       data_port("result", "Result.", any_schema()), data_port("error", "Error.", string_schema(""))},
      {state_field("enabled", "Enabled", "Enable output.", boolean_schema(true), "rw", true, true),
       state_field("connectionKey", "Connection Key", "Connection key.", string_schema(""), "rw", true, true),
       state_field("baseUrl", "Base URL", "Base URL.", string_schema("https://www.handyfeeling.com")),
       state_field("ensureHdspMode", "Ensure HDSP Mode", "Ensure HDSP mode.", boolean_schema(true)),
       state_field("invert", "Invert", "Invert position.", boolean_schema(false)),
       state_field("minPercent", "Min Percent", "Minimum percent.", number_schema(0.0, 0.0, 100.0)),
       state_field("maxPercent", "Max Percent", "Maximum percent.", number_schema(100.0, 0.0, 100.0)),
       state_field("defaultDurationMs", "Default Duration (ms)", "Default duration.", integer_schema(100, 1)),
       state_field("requestTimeoutMs", "Request Timeout (ms)", "Request timeout.", integer_schema(2000, 1)),
       state_field("minSendIntervalMs", "Min Send Interval (ms)", "Minimum send interval.", integer_schema(20, 0)),
       state_field("immediateResponse", "Immediate Response", "Immediate response.", boolean_schema(false)),
       state_field("stopOnTarget", "Stop On Target", "Stop on target.", boolean_schema(false))},
      {"exec"});
}

}  // namespace

void register_handy_out_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, handy_out_spec());
}

}  // namespace f8::cppengine
