#pragma once

#include <string>

namespace f8::cppsdk {

// Ensure a string is safe to use as a single NATS subject token (no dots).
std::string ensure_token(std::string value, const char* label);

std::string kv_bucket_for_service(const std::string& service_id);
std::string svc_micro_name(const std::string& service_id);
std::string kv_key_rungraph();
std::string kv_key_ready();
std::string kv_key_node_state(const std::string& node_id, const std::string& field);

std::string data_subject(const std::string& from_service_id, const std::string& from_node_id, const std::string& port_id);
std::string cmd_channel_subject(const std::string& service_id);
std::string svc_endpoint_subject(const std::string& service_id, const std::string& endpoint);

}  // namespace f8::cppsdk
