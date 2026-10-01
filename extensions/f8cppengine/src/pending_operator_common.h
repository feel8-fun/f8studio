#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using json = nlohmann::json;

json pending_operator_spec(const std::string& operator_class, const std::string& label, const std::string& category,
                           const std::vector<json>& data_in, const std::vector<json>& data_out,
                           const std::vector<json>& states, const std::vector<std::string>& exec_in = {},
                           const std::vector<std::string>& exec_out = {}, const json& edit_policy = nullptr);
void register_pending_operator_spec(f8::cppsdk::RuntimeNodeRegistry& registry, const json& spec);

}  // namespace f8::cppengine
