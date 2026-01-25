#include "f8cppsdk/data_bus.h"

#include <nlohmann/json.hpp>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/nats_client.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppsdk {

using json = nlohmann::json;

bool publish_data(NatsClient& nats, const std::string& from_service_id, const std::string& from_node_id,
                  const std::string& port_id, const json& value, std::int64_t ts_ms) {
  const std::int64_t ts = ts_ms > 0 ? ts_ms : now_ms();
  json payload;
  payload["value"] = value;
  payload["ts"] = ts;
  const auto bytes = payload.dump();
  const auto subject = data_subject(from_service_id, from_node_id, port_id);
  return nats.publish(subject, bytes.data(), bytes.size());
}

}  // namespace f8::cppsdk

