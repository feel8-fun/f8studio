#include "f8cppsdk/f8_naming.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace f8::cppsdk {

std::string ensure_token(std::string value, const char* label) {
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              value.end());
  if (value.empty()) {
    throw std::invalid_argument(std::string(label ? label : "token") + " must be non-empty");
  }
  if (value.find('.') != std::string::npos) {
    throw std::invalid_argument(std::string(label ? label : "token") + " must not contain '.'");
  }
  return value;
}

std::string kv_bucket_for_service(const std::string& service_id) {
  return std::string("svc_") + ensure_token(service_id, "service_id");
}

std::string kv_key_rungraph() { return "rungraph"; }
std::string kv_key_ready() { return "ready"; }

std::string kv_key_node_state(const std::string& node_id, const std::string& field) {
  const auto nid = ensure_token(node_id, "node_id");
  std::string f = field;
  f.erase(f.begin(), std::find_if(f.begin(), f.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  f.erase(std::find_if(f.rbegin(), f.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), f.end());
  if (f.empty()) {
    throw std::invalid_argument("field must be non-empty");
  }
  return "nodes." + nid + ".state." + f;
}

std::string data_subject(const std::string& from_service_id, const std::string& from_node_id, const std::string& port_id) {
  return "svc." + ensure_token(from_service_id, "from_service_id") + ".nodes." + ensure_token(from_node_id, "from_node_id") +
         ".data." + ensure_token(port_id, "port_id");
}

std::string cmd_channel_subject(const std::string& service_id) {
  return "svc." + ensure_token(service_id, "service_id") + ".cmd";
}

std::string svc_endpoint_subject(const std::string& service_id, const std::string& endpoint) {
  return "svc." + ensure_token(service_id, "service_id") + "." + ensure_token(endpoint, "endpoint");
}

}  // namespace f8::cppsdk

