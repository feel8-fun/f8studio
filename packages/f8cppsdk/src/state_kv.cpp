#include "f8cppsdk/state_kv.h"

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppsdk {

using json = nlohmann::json;

bool kv_set_ready(KvStore& kv, const std::string& service_id, bool ready, const std::string& reason,
                  std::int64_t ts_ms) {
  const std::int64_t ts = ts_ms > 0 ? ts_ms : now_ms();
  json payload = json::object();
  payload["serviceId"] = ensure_token(service_id, "service_id");
  payload["ready"] = ready;
  payload["reason"] = reason;
  payload["ts"] = ts;
  const auto raw = payload.dump();
  return kv.put(kv_key_ready(), raw.data(), raw.size());
}

bool kv_set_node_state(KvStore& kv, const std::string& service_id, const std::string& node_id, const std::string& field,
                       const json& value, const std::string& source, const json& extra_meta, std::int64_t ts_ms,
                       const std::string& origin) {
  const std::int64_t ts = ts_ms > 0 ? ts_ms : now_ms();
  json payload;
  payload["value"] = value;
  payload["actor"] = ensure_token(service_id, "service_id");
  payload["ts"] = ts;
  if (!source.empty()) {
    payload["source"] = source;
  }
  if (!origin.empty()) {
    payload["origin"] = origin;
  }
  if (extra_meta.is_object()) {
    for (auto it = extra_meta.begin(); it != extra_meta.end(); ++it) {
      const std::string k = it.key();
      if (k == "value" || k == "actor" || k == "ts" || k == "source" || k == "origin") {
        continue;
      }
      payload[k] = it.value();
    }
  }

  const auto key = kv_key_node_state(node_id, field);
  const auto raw = payload.dump();
  return kv.put(key, raw.data(), raw.size());
}

}  // namespace f8::cppsdk
