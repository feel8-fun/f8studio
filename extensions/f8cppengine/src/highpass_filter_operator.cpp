#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json highpass_filter_spec() {
  return pending_operator_spec(
      "f8.highpass_filter", "Highpass Filter", "signal",
      {data_port("value", "Value to filter.", any_schema())}, {data_port("value", "Filtered output.", any_schema())},
      {state_field("sampleIntervalMs", "Sample Interval (ms)", "Sampling interval in milliseconds.", number_schema(1000.0 / 120.0, 0.001, 50000.0), "rw", true, true),
       state_field("cutoff", "Cutoff", "High-pass cutoff frequency in Hz.", number_schema(1.0, 0.001, 5000.0), "rw", true, true),
       state_field("order", "Order", "Butterworth filter order.", number_schema(2.0, 1.0, 12.0)),
       state_field("reset_on_state_change", "Reset On State Change", "Reset filter history when parameters change.", boolean_schema(true))});
}

}  // namespace

void register_highpass_filter_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, highpass_filter_spec());
}

}  // namespace f8::cppengine
