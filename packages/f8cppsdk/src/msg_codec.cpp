#include "f8cppsdk/msg_codec.h"

#include <exception>

namespace f8::cppsdk {

std::vector<std::uint8_t> encode_json(const nlohmann::json& value) {
  return nlohmann::json::to_msgpack(value);
}

bool decode_json(const void* data, std::size_t len, nlohmann::json& out) {
  if (data == nullptr || len == 0) {
    return false;
  }
  try {
    const auto* begin = static_cast<const std::uint8_t*>(data);
    const auto* end = begin + len;
    out = nlohmann::json::from_msgpack(begin, end);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace f8::cppsdk
