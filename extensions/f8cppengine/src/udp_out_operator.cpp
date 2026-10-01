#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json udp_out_spec() {
  return pending_operator_spec(
      "f8.udp_out", "UDP Out", "output", {data_port("value", "Value to send.", any_schema())},
      {data_port("isOpen", "Socket open.", boolean_schema(false)), data_port("sentBytes", "Sent bytes.", integer_schema(0, 0)),
       data_port("error", "Error.", string_schema(""))},
      {state_field("enabled", "Enabled", "Enable UDP output.", boolean_schema(true), "rw", true, true),
       state_field("host", "Host", "Destination host.", string_schema("127.0.0.1"), "rw", true, true),
       state_field("port", "Port", "Destination port.", integer_schema(9000, 0, 65535), "rw", true, true),
       state_field("appendNewline", "Append Newline", "Append newline.", boolean_schema(false)),
       state_field("forceText", "Force Text", "Force text encoding.", boolean_schema(false))},
      {"exec"});
}

}  // namespace

void register_udp_out_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, udp_out_spec());
}

}  // namespace f8::cppengine
