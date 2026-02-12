#include "f8cvkit/base64.h"

#include <array>
#include <cctype>
#include <limits>
#include <string>

namespace f8::cvkit {

namespace {

constexpr std::uint8_t kInvalid = 0xFF;
constexpr char kB64Alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::array<std::uint8_t, 256> make_table() {
  std::array<std::uint8_t, 256> t{};
  t.fill(kInvalid);
  for (int i = 0; i < 26; ++i) {
    t[static_cast<std::size_t>('A' + i)] = static_cast<std::uint8_t>(i);
    t[static_cast<std::size_t>('a' + i)] = static_cast<std::uint8_t>(26 + i);
  }
  for (int i = 0; i < 10; ++i) {
    t[static_cast<std::size_t>('0' + i)] = static_cast<std::uint8_t>(52 + i);
  }
  t[static_cast<std::size_t>('+')] = 62;
  t[static_cast<std::size_t>('/')] = 63;
  return t;
}

const std::array<std::uint8_t, 256> kTable = make_table();

}  // namespace

std::string base64_encode(const std::uint8_t* data, std::size_t len) {
  if (data == nullptr || len == 0) {
    return {};
  }

  const std::size_t full = len / 3;
  const std::size_t rem = len % 3;
  const std::size_t out_len = (full * 4) + (rem == 0 ? 0 : 4);

  std::string out;
  out.resize(out_len);

  std::size_t i = 0;
  std::size_t o = 0;
  for (std::size_t b = 0; b < full; ++b) {
    const std::uint32_t x = (static_cast<std::uint32_t>(data[i + 0]) << 16) |
                            (static_cast<std::uint32_t>(data[i + 1]) << 8) |
                            (static_cast<std::uint32_t>(data[i + 2]) << 0);
    i += 3;

    out[o + 0] = kB64Alphabet[(x >> 18) & 0x3Fu];
    out[o + 1] = kB64Alphabet[(x >> 12) & 0x3Fu];
    out[o + 2] = kB64Alphabet[(x >> 6) & 0x3Fu];
    out[o + 3] = kB64Alphabet[x & 0x3Fu];
    o += 4;
  }

  if (rem == 1) {
    const std::uint32_t x = (static_cast<std::uint32_t>(data[i + 0]) << 16);
    out[o + 0] = kB64Alphabet[(x >> 18) & 0x3Fu];
    out[o + 1] = kB64Alphabet[(x >> 12) & 0x3Fu];
    out[o + 2] = '=';
    out[o + 3] = '=';
  } else if (rem == 2) {
    const std::uint32_t x = (static_cast<std::uint32_t>(data[i + 0]) << 16) |
                            (static_cast<std::uint32_t>(data[i + 1]) << 8);
    out[o + 0] = kB64Alphabet[(x >> 18) & 0x3Fu];
    out[o + 1] = kB64Alphabet[(x >> 12) & 0x3Fu];
    out[o + 2] = kB64Alphabet[(x >> 6) & 0x3Fu];
    out[o + 3] = '=';
  }

  return out;
}

std::string base64_encode(const std::vector<std::uint8_t>& bytes) {
  return base64_encode(bytes.data(), bytes.size());
}

Base64DecodeResult base64_decode(const std::string& b64) {
  Base64DecodeResult out;
  if (b64.empty()) {
    return out;
  }

  std::vector<std::uint8_t> clean;
  clean.reserve(b64.size());
  for (unsigned char ch : b64) {
    if (std::isspace(ch)) continue;
    clean.push_back(static_cast<std::uint8_t>(ch));
  }

  if (clean.size() % 4 != 0) {
    out.error = "invalid base64 length";
    return out;
  }

  out.bytes.reserve((clean.size() / 4) * 3);

  for (std::size_t i = 0; i < clean.size(); i += 4) {
    const std::uint8_t c0 = clean[i + 0];
    const std::uint8_t c1 = clean[i + 1];
    const std::uint8_t c2 = clean[i + 2];
    const std::uint8_t c3 = clean[i + 3];

    const bool pad2 = (c2 == '=');
    const bool pad3 = (c3 == '=');

    const std::uint8_t v0 = kTable[c0];
    const std::uint8_t v1 = kTable[c1];
    const std::uint8_t v2 = pad2 ? 0 : kTable[c2];
    const std::uint8_t v3 = pad3 ? 0 : kTable[c3];

    if (v0 == kInvalid || v1 == kInvalid) {
      out.error = "invalid base64 character";
      out.bytes.clear();
      return out;
    }
    if (!pad2 && v2 == kInvalid) {
      out.error = "invalid base64 character";
      out.bytes.clear();
      return out;
    }
    if (!pad3 && v3 == kInvalid) {
      out.error = "invalid base64 character";
      out.bytes.clear();
      return out;
    }
    if (pad2 && !pad3) {
      out.error = "invalid base64 padding";
      out.bytes.clear();
      return out;
    }

    const std::uint32_t x = (static_cast<std::uint32_t>(v0) << 18) | (static_cast<std::uint32_t>(v1) << 12) |
                            (static_cast<std::uint32_t>(v2) << 6) | static_cast<std::uint32_t>(v3);

    out.bytes.push_back(static_cast<std::uint8_t>((x >> 16) & 0xFFu));
    if (!pad2) {
      out.bytes.push_back(static_cast<std::uint8_t>((x >> 8) & 0xFFu));
    }
    if (!pad3) {
      out.bytes.push_back(static_cast<std::uint8_t>(x & 0xFFu));
    }
  }

  return out;
}

}  // namespace f8::cvkit
