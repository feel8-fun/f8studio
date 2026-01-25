#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace f8::cppsdk {

class KvStore;

// KV helpers compatible with f8pysdk payload conventions.
bool kv_set_ready(KvStore& kv, bool ready, std::int64_t ts_ms = 0);

bool kv_set_node_state(KvStore& kv, const std::string& service_id, const std::string& node_id, const std::string& field,
                       const nlohmann::json& value, const std::string& source = "runtime",
                       const nlohmann::json& extra_meta = nlohmann::json::object(), std::int64_t ts_ms = 0);

}  // namespace f8::cppsdk
