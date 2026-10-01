#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json udp_in_spec() {
  return pending_operator_spec(
      "f8.udp_in", "UDP In", "io", {}, {data_port("text", "Text payload.", string_schema("")),
                                         data_port("raw", "Raw payload.", any_schema()),
                                         data_port("json", "JSON payload.", any_schema()),
                                         data_port("packet", "Packet.", any_schema())},
      {state_field("bindAddress", "Bind Address", "Bind address.", string_schema("127.0.0.1")),
       state_field("allowNonLoopbackBind", "Allow Non-loopback Bind", "Allow non-loopback bind.", boolean_schema(false)),
       state_field("port", "Port", "UDP port.", integer_schema(9000, 0, 65535)),
       state_field("maxQueue", "Max Queue", "Maximum queue length.", integer_schema(1024, 1)),
       state_field("reuseAddress", "Reuse Address", "Reuse address.", boolean_schema(true)),
       state_field("listening", "Listening", "Listening.", boolean_schema(false), "ro", true, true)},
      {}, {"packet"});
}

}  // namespace

void register_udp_in_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, udp_in_spec());
}

}  // namespace f8::cppengine
