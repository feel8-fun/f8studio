#include "f8cppengine/operators.h"

#include "operator_modules.h"

namespace f8::cppengine {

void register_cppengine_specs(f8::cppsdk::RuntimeNodeRegistry& registry) {
  register_native_operator_specs(registry);
  register_script_operator_specs(registry);
}

}  // namespace f8::cppengine