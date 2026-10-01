#include "native_operator_modules.h"

namespace f8::cppengine {

void register_native_operator_specs(f8::cppsdk::RuntimeNodeRegistry& registry) {
  register_cppengine_service_node(registry);
  register_tick_operator(registry);
  register_range_map_operator(registry);
  register_data_expr_operator(registry);
  register_data_pick_operator(registry);
  register_exec_sequence_operator(registry);
  register_print_operator(registry);
  register_phase_operator(registry);
  register_cosine_operator(registry);
  register_tempest_operator(registry);
  register_smooth_filter_operator(registry);
  register_detrend_operator(registry);
  register_rate_limiter_operator(registry);
  register_tcode_operator(registry);
  register_quat_to_euler_operator(registry);
  register_silence_detector_operator(registry);
  register_switch_mixer_operator(registry);
  register_unsupported_operator_specs(registry);
}

}  // namespace f8::cppengine
