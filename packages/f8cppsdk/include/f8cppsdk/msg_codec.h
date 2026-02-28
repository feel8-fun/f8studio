#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace f8::cppsdk {

std::vector<std::uint8_t> encode_json(const nlohmann::json& value);
bool decode_json(const void* data, std::size_t len, nlohmann::json& out);

}  // namespace f8::cppsdk
