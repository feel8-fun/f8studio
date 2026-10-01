#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace f8::cvkit {

// Encode raw bytes as standard base64 (RFC 4648). No line breaks.
std::string base64_encode(const std::uint8_t* data, std::size_t len);
std::string base64_encode(const std::vector<std::uint8_t>& bytes);

struct Base64DecodeResult {
  std::vector<std::uint8_t> bytes;
  std::string error;
};

// Decode base64 into raw bytes. Accepts standard base64 and ignores whitespace.
Base64DecodeResult base64_decode(const std::string& b64);

}  // namespace f8::cvkit
