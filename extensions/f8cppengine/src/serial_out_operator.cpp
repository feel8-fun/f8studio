#include "pending_operator_common.h"

#include "operator_common.h"

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::RuntimeNodeRegistry;

namespace {

json serial_out_spec() {
  return pending_operator_spec(
      "f8.serial_out", "Serial Out", "output", {data_port("value", "Value to send.", any_schema())},
      {data_port("isOpen", "Whether port is open.", boolean_schema(false)), data_port("writtenBytes", "Written bytes.", integer_schema(0, 0)),
       data_port("error", "Error.", string_schema(""))},
      {state_field("enabled", "Enabled", "Enable serial output.", boolean_schema(true), "rw", true, true),
       state_field("port", "Port", "Serial port.", string_schema(""), "rw", true, true),
       state_field("baudrate", "Baudrate", "Baud rate.", integer_schema(115200, 1), "rw", true, true)},
      {"exec"});
}

}  // namespace

void register_serial_out_operator(RuntimeNodeRegistry& registry) {
  register_pending_operator_spec(registry, serial_out_spec());
}

}  // namespace f8::cppengine
