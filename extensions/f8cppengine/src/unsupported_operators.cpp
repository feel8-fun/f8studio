#include "native_operator_modules.h"

namespace f8::cppengine {

void register_unsupported_operator_specs(f8::cppsdk::RuntimeNodeRegistry& registry) {
  register_flow_motion_pending_operators(registry);
  register_bandpass_filter_operator(registry);
  register_highpass_filter_operator(registry);
  register_lowpass_filter_operator(registry);
  register_envelope_operator(registry);
  register_periodicity_detector_operator(registry);
  register_bone_filter_operator(registry);
  register_bone_selector_operator(registry);
  register_playback_sync_operator(registry);
  register_program_wave_operator(registry);
  register_sequence_player_operator(registry);
  register_wave_expr_operator(registry);
  register_wave_pattern_operator(registry);
  register_wave_funscript_operator(registry);
  register_skeleton_decoder_operator(registry);
  register_vmc_decoder_operator(registry);
  register_state_trigger_operator(registry);
  register_state_expr_operator(registry);
  register_recorder_operator(registry);
  register_replayer_operator(registry);
  register_serial_out_operator(registry);
  register_udp_in_operator(registry);
  register_udp_out_operator(registry);
  register_handy_out_operator(registry);
  register_lovense_out_operator(registry);
  register_buttplug_out_operator(registry);
  register_lovense_mock_server_operator(registry);
}

}  // namespace f8::cppengine
