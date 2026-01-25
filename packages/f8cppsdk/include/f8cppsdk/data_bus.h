#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json_fwd.hpp>

namespace f8::cppsdk {

class NatsClient;

// Publish a data sample on `svc.<fromServiceId>.nodes.<fromNodeId>.data.<portId>`.
// Payload format matches f8pysdk ServiceBus.emit_data: {"value": <json>, "ts": <ms>}.
bool publish_data(NatsClient& nats, const std::string& from_service_id, const std::string& from_node_id,
                  const std::string& port_id, const nlohmann::json& value, std::int64_t ts_ms = 0);

}  // namespace f8::cppsdk
