#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json lovense_mock_server_spec() {
  return pending_operator_spec(
      "f8.lovense_mock_server", "Lovense Mock Server", "io", {}, {data_port("event", "Event.", any_schema())},
      {state_field("bindAddress", "Bind Address", "Bind address.", string_schema("127.0.0.1")),
       state_field("allowNonLoopbackBind", "Allow Non-loopback Bind", "Allow non-loopback bind.", boolean_schema(false)),
       state_field("port", "Port", "Port.", integer_schema(30010, 0, 65535)),
       state_field("printEnabled", "Print Enabled", "Print events.", boolean_schema(true)),
       state_field("eventIncludePayload", "Event Include Payload", "Include payload.", boolean_schema(true)),
       state_field("eventIncludeRequest", "Event Include Request", "Include request.", boolean_schema(false)),
       state_field("listening", "Listening", "Listening.", boolean_schema(false), "ro")},
      {}, {"event"});
}

}  // namespace

void register_lovense_mock_server_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, lovense_mock_server_spec());
}

}  // namespace f8::cppengine
