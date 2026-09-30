#include "f8cppsdk/zenoh_naming.h"
#include "f8cppsdk/generated/runtime_keys.h"

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/runtime_backend.h"

#include <stdexcept>
#include <string_view>
#include <vector>

namespace f8::cppsdk {
namespace {

constexpr const char* kF8Prefix = "f8";

std::vector<std::string> split_non_empty(std::string_view value, char sep) {
  std::vector<std::string> parts;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(sep, begin);
    const std::size_t limit = end == std::string_view::npos ? value.size() : end;
    const std::string part = trim_runtime_string(value.substr(begin, limit - begin));
    if (!part.empty()) {
      parts.push_back(part);
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return parts;
}

std::string join_parts(const std::vector<std::string>& parts, char sep) {
  std::string out;
  for (const std::string& part : parts) {
    if (!out.empty()) {
      out.push_back(sep);
    }
    out += part;
  }
  return out;
}

std::string field_to_path(const std::string& field) {
  const auto parts = split_non_empty(field, '.');
  if (parts.empty()) {
    throw std::invalid_argument("field must be non-empty");
  }
  return join_parts(parts, '/');
}

std::string path_to_field(const std::string& path) {
  const auto parts = split_non_empty(path, '/');
  if (parts.empty()) {
    throw std::invalid_argument("field path must be non-empty");
  }
  return join_parts(parts, '.');
}

std::optional<std::pair<std::string, std::string>> parse_state_path_node_field(const std::string& key) {
  const std::string text = trim_runtime_string(key);
  constexpr const char* kPrefix = "nodes.";
  constexpr const char* kStateMarker = ".state.";
  if (text.rfind(kPrefix, 0) != 0) {
    return std::nullopt;
  }
  const std::size_t marker = text.find(kStateMarker);
  if (marker == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t node_begin = std::string_view(kPrefix).size();
  if (marker <= node_begin) {
    return std::nullopt;
  }
  const std::string node_id = text.substr(node_begin, marker - node_begin);
  const std::size_t field_begin = marker + std::string_view(kStateMarker).size();
  if (field_begin >= text.size()) {
    return std::nullopt;
  }
  return std::make_pair(node_id, text.substr(field_begin));
}

}  // namespace

std::string zenoh_data_key(const std::string& service_id, const std::string& node_id, const std::string& port_id) {
  return wire_keys::data(ensure_token(service_id, "service_id"), ensure_token(node_id, "node_id"), ensure_token(port_id, "port_id"));
}

std::string zenoh_command_key(const std::string& service_id, const std::string& command) {
  return wire_keys::command(ensure_token(service_id, "service_id"), ensure_token(command, "command"));
}

std::string zenoh_service_liveliness_key(const std::string& service_id, const std::string& runtime_instance_id) {
  return wire_keys::service_liveliness(ensure_token(service_id, "service_id"), ensure_token(runtime_instance_id, "runtime_instance_id"));
}

std::string zenoh_studio_liveliness_key(const std::string& studio_service_id) {
  return wire_keys::studio_liveliness(ensure_token(studio_service_id, "studio_service_id"));
}

std::string zenoh_state_key(const std::string& service_id, const std::string& node_id, const std::string& field) {
  return wire_keys::node_state(ensure_token(service_id, "service_id"), ensure_token(node_id, "node_id"), field_to_path(field));
}

std::string zenoh_state_path_key(const std::string& service_id, const std::string& key) {
  const auto parsed = parse_state_path_node_field(key);
  if (parsed.has_value()) {
    return zenoh_state_key(service_id, parsed->first, parsed->second);
  }

  const auto parts = split_non_empty(key, '.');
  if (parts.empty()) {
    throw std::invalid_argument("key must be non-empty");
  }
  return std::string(kF8Prefix) + "/svc/" + ensure_token(service_id, "service_id") + "/state/" + join_parts(parts, '/');
}

std::string zenoh_state_path_pattern(const std::string& service_id, const std::string& key_pattern) {
  const std::string pattern = trim_runtime_string(key_pattern);
  if (pattern.empty()) {
    throw std::invalid_argument("key_pattern must be non-empty");
  }
  if (!pattern.empty() && pattern.back() == '>') {
    std::string prefix = trim_runtime_string(pattern.substr(0, pattern.size() - 1));
    while (!prefix.empty() && prefix.back() == '.') {
      prefix.pop_back();
    }
    const auto parts = split_non_empty(prefix, '.');
    if (!parts.empty() && parts[0] == "nodes") {
      const std::string sid = ensure_token(service_id, "service_id");
      if (parts.size() == 1) {
        return std::string(kF8Prefix) + "/svc/" + sid + "/state/nodes/**";
      }
      const std::string node_id = ensure_token(parts[1], "node_id");
      if (parts.size() == 2) {
        return std::string(kF8Prefix) + "/svc/" + sid + "/state/nodes/" + node_id + "/**";
      }
      if (parts.size() >= 3 && parts[2] == "state") {
        return std::string(kF8Prefix) + "/svc/" + sid + "/state/nodes/" + node_id + "/state/**";
      }
    }
    const std::string path = join_parts(parts, '/');
    return std::string(kF8Prefix) + "/svc/" + ensure_token(service_id, "service_id") + "/state/" + path + "/**";
  }
  return zenoh_state_path_key(service_id, pattern);
}

std::optional<std::string> zenoh_key_to_state_path(const std::string& key) {
  const auto parts = split_non_empty(key, '/');
  if (parts.size() >= 8 && parts[0] == kF8Prefix && parts[1] == "svc" && parts[3] == "state") {
    if (parts[4] != "nodes" || parts[6] != "state") {
      return std::nullopt;
    }
    std::vector<std::string> field_parts(parts.begin() + 7, parts.end());
    return std::string("nodes.") + parts[5] + ".state." + path_to_field(join_parts(field_parts, '/'));
  }
  if (parts.size() >= 5 && parts[0] == kF8Prefix && parts[1] == "svc" && parts[3] == "state") {
    std::vector<std::string> key_parts(parts.begin() + 4, parts.end());
    return join_parts(key_parts, '.');
  }
  return std::nullopt;
}

}  // namespace f8::cppsdk
