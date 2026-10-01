#pragma once

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

void register_cppengine_service_node(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_tick_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_range_map_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_data_expr_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_data_pick_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_exec_sequence_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_flow_motion_pending_operators(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_print_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_phase_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_cosine_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_tempest_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_smooth_filter_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_detrend_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_rate_limiter_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_tcode_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_quat_to_euler_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_silence_detector_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_switch_mixer_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_bandpass_filter_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_highpass_filter_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_lowpass_filter_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_envelope_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_periodicity_detector_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_bone_filter_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_bone_selector_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_playback_sync_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_program_wave_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_sequence_player_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_wave_expr_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_wave_pattern_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_wave_funscript_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_skeleton_decoder_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_vmc_decoder_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_state_trigger_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_state_expr_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_recorder_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_replayer_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_serial_out_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_udp_in_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_udp_out_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_handy_out_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_lovense_out_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_buttplug_out_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_lovense_mock_server_operator(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_unsupported_operator_specs(f8::cppsdk::RuntimeNodeRegistry& registry);

}  // namespace f8::cppengine
