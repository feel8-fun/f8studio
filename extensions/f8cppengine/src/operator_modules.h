#pragma once

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

void register_native_operator_specs(f8::cppsdk::RuntimeNodeRegistry& registry);
void register_script_operator_specs(f8::cppsdk::RuntimeNodeRegistry& registry);

}  // namespace f8::cppengine
