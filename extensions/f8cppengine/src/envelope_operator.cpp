#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json envelope_spec() {
  return pending_operator_spec(
      "f8.envelope", "Envelope", "signal", {data_port("value", "Value input.", any_schema())},
      {data_port("lower", "Lower envelope.", any_schema()), data_port("upper", "Upper envelope.", any_schema()),
       data_port("normalized", "Normalized output.", any_schema())},
      {state_field("method", "Method", "Envelope method.", string_enum_schema("EMA", {"EMA", "SMA"}), "rw", true, true),
       state_field("rise_alpha", "Rise Alpha", "Rise alpha.", number_schema(0.2, 0.0, 1.0), "rw", true, true),
       state_field("fall_alpha", "Fall Alpha", "Fall alpha.", number_schema(0.05, 0.0, 1.0), "rw", true, true),
       state_field("min_span", "Min Span", "Minimum envelope span.", number_schema(0.001, 0.0)),
       state_field("sma_window", "SMA Window", "Simple moving average window.", integer_schema(30, 1, 10000)),
       state_field("margin", "Margin", "Envelope margin.", number_schema(0.0, 0.0)),
       state_field("jumpEnabled", "Jump Enabled", "Enable jump reseed.", boolean_schema(true)),
       state_field("jumpSpanMult", "Jump Span Mult", "Jump span multiplier.", number_schema(4.0, 0.0)),
       state_field("jumpConsecutiveFrames", "Jump Consecutive Frames", "Frames before jump reseed.", integer_schema(3, 1)),
       state_field("jumpReseedFrames", "Jump Reseed Frames", "Frames used for reseed.", integer_schema(6, 1))});
}

}  // namespace

void register_envelope_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, envelope_spec());
}

}  // namespace f8::cppengine
