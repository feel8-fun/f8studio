#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/generated/runtime_keys.h"

#include <array>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace f8::cppsdk {

namespace {

std::uint32_t rotr32(std::uint32_t value, std::uint32_t bits) {
  return (value >> bits) | (value << (32U - bits));
}

std::string sha256_hex(const std::string& input) {
  static constexpr std::array<std::uint32_t, 64> k = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
      0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
      0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
      0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
      0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
      0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
      0xc67178f2U};

  std::vector<std::uint8_t> data(input.begin(), input.end());
  const std::uint64_t bit_len = static_cast<std::uint64_t>(data.size()) * 8U;
  data.push_back(0x80U);
  while ((data.size() % 64U) != 56U) {
    data.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    data.push_back(static_cast<std::uint8_t>((bit_len >> shift) & 0xffU));
  }

  std::uint32_t h0 = 0x6a09e667U;
  std::uint32_t h1 = 0xbb67ae85U;
  std::uint32_t h2 = 0x3c6ef372U;
  std::uint32_t h3 = 0xa54ff53aU;
  std::uint32_t h4 = 0x510e527fU;
  std::uint32_t h5 = 0x9b05688cU;
  std::uint32_t h6 = 0x1f83d9abU;
  std::uint32_t h7 = 0x5be0cd19U;

  for (std::size_t offset = 0; offset < data.size(); offset += 64U) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16U; ++i) {
      const std::size_t j = offset + i * 4U;
      w[i] = (static_cast<std::uint32_t>(data[j]) << 24U) |
             (static_cast<std::uint32_t>(data[j + 1U]) << 16U) |
             (static_cast<std::uint32_t>(data[j + 2U]) << 8U) |
             static_cast<std::uint32_t>(data[j + 3U]);
    }
    for (std::size_t i = 16U; i < 64U; ++i) {
      const std::uint32_t s0 = rotr32(w[i - 15U], 7U) ^ rotr32(w[i - 15U], 18U) ^ (w[i - 15U] >> 3U);
      const std::uint32_t s1 = rotr32(w[i - 2U], 17U) ^ rotr32(w[i - 2U], 19U) ^ (w[i - 2U] >> 10U);
      w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }

    std::uint32_t a = h0;
    std::uint32_t b = h1;
    std::uint32_t c = h2;
    std::uint32_t d = h3;
    std::uint32_t e = h4;
    std::uint32_t f = h5;
    std::uint32_t g = h6;
    std::uint32_t h = h7;

    for (std::size_t i = 0; i < 64U; ++i) {
      const std::uint32_t s1 = rotr32(e, 6U) ^ rotr32(e, 11U) ^ rotr32(e, 25U);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr32(a, 2U) ^ rotr32(a, 13U) ^ rotr32(a, 22U);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
    h5 += f;
    h6 += g;
    h7 += h;
  }

  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0');
  for (const auto word : {h0, h1, h2, h3, h4, h5, h6, h7}) {
    out << std::setw(8) << word;
  }
  return out.str();
}

}  // namespace

std::string ensure_token(std::string value, const char* label) {
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              value.end());
  if (value.empty()) {
    throw std::invalid_argument(std::string(label ? label : "token") + " must be non-empty");
  }
  if (value.find('.') != std::string::npos || value.find('/') != std::string::npos) {
    throw std::invalid_argument(std::string(label ? label : "token") + " must not contain '.' or '/'");
  }
  return value;
}

std::string rungraph_key(const std::string& service_id) {
  return wire_keys::rungraph(ensure_token(service_id, "service_id"));
}

std::string rungraph_deploy_status_key(const std::string& service_id) {
  return wire_keys::rungraph_status(ensure_token(service_id, "service_id"));
}

std::string rungraph_deploy_request_status_key(const std::string& service_id, const std::string& req_id) {
  std::string trimmed_req_id = req_id;
  trimmed_req_id.erase(trimmed_req_id.begin(),
                       std::find_if(trimmed_req_id.begin(), trimmed_req_id.end(),
                                    [](unsigned char ch) { return !std::isspace(ch); }));
  trimmed_req_id.erase(std::find_if(trimmed_req_id.rbegin(), trimmed_req_id.rend(),
                                    [](unsigned char ch) { return !std::isspace(ch); })
                           .base(),
                       trimmed_req_id.end());
  if (trimmed_req_id.empty()) {
    throw std::invalid_argument("req_id must be non-empty");
  }
  return rungraph_deploy_status_key(service_id) + "/requests/" + sha256_hex(trimmed_req_id);
}

std::string ready_key(const std::string& service_id) {
  return wire_keys::ready(ensure_token(service_id, "service_id"));
}

std::string state_path_node_field(const std::string& node_id, const std::string& field) {
  const auto nid = ensure_token(node_id, "node_id");
  std::string f = field;
  f.erase(f.begin(), std::find_if(f.begin(), f.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  f.erase(std::find_if(f.rbegin(), f.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), f.end());
  if (f.empty()) {
    throw std::invalid_argument("field must be non-empty");
  }
  return "nodes." + nid + ".state." + f;
}

std::string data_key(const std::string& from_service_id, const std::string& from_node_id,
                     const std::string& port_id) {
  return wire_keys::data(ensure_token(from_service_id, "from_service_id"), ensure_token(from_node_id, "from_node_id"), ensure_token(port_id, "port_id"));
}

std::string cmd_channel_key(const std::string& service_id) {
  return wire_keys::command(ensure_token(service_id, "service_id"), "cmd");
}

std::string svc_endpoint_key(const std::string& service_id, const std::string& endpoint) {
  return wire_keys::command(ensure_token(service_id, "service_id"), ensure_token(endpoint, "endpoint"));
}

}  // namespace f8::cppsdk
