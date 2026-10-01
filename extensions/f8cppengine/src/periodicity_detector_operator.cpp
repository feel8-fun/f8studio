#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json periodicity_detector_spec() {
  return pending_operator_spec(
      "f8.periodicity_detector", "Periodicity Detector", "analysis",
      {data_port("value", "Scalar signal input.", number_schema())},
      {data_port("confidence", "Periodicity confidence.", number_schema()), data_port("rms", "RMS level.", number_schema()),
       data_port("periodicEnergy", "Periodic energy.", number_schema()), data_port("periodMs", "Period in milliseconds.", number_schema()),
       data_port("period_hz", "Period in Hz.", number_schema()), data_port("is_periodic", "Whether periodic.", boolean_schema(false))},
      {state_field("window", "Window", "Analysis window.", integer_schema(240, 1)),
       state_field("min_lag", "Min Lag", "Minimum lag.", integer_schema(4, 1)),
       state_field("max_lag", "Max Lag", "Maximum lag.", integer_schema(120, 1)),
       state_field("peak_prominence", "Peak Prominence", "Peak prominence.", number_schema(0.1, 0.0)),
       state_field("min_peaks", "Min Peaks", "Minimum peaks.", integer_schema(1, 1)),
       state_field("smoothing_alpha", "Smoothing Alpha", "Smoothing alpha.", number_schema(0.3, 0.0, 1.0)),
       state_field("noise_floor", "Noise Floor", "Noise floor.", number_schema(0.001, 0.0)),
       state_field("threshold", "Threshold", "Periodicity threshold.", number_schema(0.5, 0.0, 1.0)),
       state_field("rms_window", "RMS Window", "RMS window.", integer_schema(60, 1)),
       state_field("sampleIntervalMs", "Sample Interval (ms)", "Sampling interval.", number_schema(1000.0 / 120.0, 0.001, 50000.0)),
       state_field("reset_on_missing", "Reset On Missing", "Reset when input is missing.", boolean_schema(true))});
}

}  // namespace

void register_periodicity_detector_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, periodicity_detector_spec());
}

}  // namespace f8::cppengine
