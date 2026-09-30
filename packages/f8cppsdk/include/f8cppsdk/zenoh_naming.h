#pragma once

#include <optional>
#include <string>

namespace f8::cppsdk {

std::string zenoh_data_key(const std::string& service_id, const std::string& node_id, const std::string& port_id);
std::string zenoh_command_key(const std::string& service_id, const std::string& command);
std::string zenoh_service_liveliness_key(const std::string& service_id, const std::string& runtime_instance_id);
std::string zenoh_studio_liveliness_key(const std::string& studio_service_id);
std::string zenoh_state_key(const std::string& service_id, const std::string& node_id, const std::string& field);
std::string zenoh_state_path_key(const std::string& service_id, const std::string& path);
std::string zenoh_state_path_pattern(const std::string& service_id, const std::string& path_pattern);
std::optional<std::string> zenoh_key_to_state_path(const std::string& key);

}  // namespace f8::cppsdk
