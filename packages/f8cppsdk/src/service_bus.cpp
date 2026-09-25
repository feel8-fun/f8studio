#include "f8cppsdk/service_bus.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/generated/protocol_models.h"
#include "f8cppsdk/msg_codec.h"
#include "f8cppsdk/rungraph_routes.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/zenoh_naming.h"
#include "f8cppsdk/zenoh_transport.h"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace f8::cppsdk {

using json = nlohmann::json;

namespace {

bool state_debug_enabled() {
  const char* v = std::getenv("F8_STATE_DEBUG");
  if (v == nullptr) return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

std::string runtime_instance_id_hex() {
  std::random_device rd;
  std::uniform_int_distribution<int> dist(0, 15);
  std::string out;
  out.reserve(32);
  static constexpr char kHex[] = "0123456789abcdef";
  for (int i = 0; i < 32; ++i) {
    out.push_back(kHex[dist(rd)]);
  }
  return out;
}

std::int64_t coerce_inbound_ts_ms(const json& payload, std::int64_t default_ts_ms) {
  auto read_int = [&](const char* key) -> std::optional<std::int64_t> {
    try {
      if (!payload.is_object() || !payload.contains(key)) return std::nullopt;
      const auto& v = payload.at(key);
      if (v.is_number_integer()) return v.get<std::int64_t>();
      if (v.is_number_float()) return static_cast<std::int64_t>(v.get<double>());
      if (v.is_string()) return std::stoll(v.get<std::string>());
    } catch (...) {
      return std::nullopt;
    }
    return std::nullopt;
  };

  std::optional<std::int64_t> ts = read_int("ts");
  if (!ts.has_value()) ts = read_int("ts_ms");
  if (!ts.has_value()) ts = read_int("tsMs");

  std::int64_t t = ts.value_or(default_ts_ms);
  if (t <= 0) return default_ts_ms;

  // Heuristics matching pysdk:
  // - seconds ~1e9, ms ~1e12 (2026), micros ~1e15, nanos ~1e18
  if (t < 100'000'000'000LL) return t * 1000LL;
  if (t >= 100'000'000'000'000'000LL) return t / 1'000'000LL;
  if (t >= 100'000'000'000'000LL) return t / 1000LL;
  return t;
}

bool state_origin_allows_access(const std::string& origin, const std::string& access) {
  // origin: "external" | "runtime" | "rungraph" | "system"
  // access: "rw" | "ro" | "wo"
  if (origin == "system") return true;
  if (origin == "runtime") return (access == "rw" || access == "ro");
  if (origin == "rungraph") return (access == "rw" || access == "wo");
  if (origin == "external") return (access == "rw" || access == "wo");
  return false;
}

std::string access_to_string(f8::cppsdk::generated::F8StateAccess a) {
  switch (a) {
    case f8::cppsdk::generated::F8StateAccess::rw:
      return "rw";
    case f8::cppsdk::generated::F8StateAccess::ro:
      return "ro";
    case f8::cppsdk::generated::F8StateAccess::wo:
      return "wo";
  }
  return "";
}

std::string trim_copy(std::string value) {
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              value.end());
  return value;
}

std::string normalize_monitor_error_severity(std::string severity) {
  severity = trim_copy(std::move(severity));
  std::transform(severity.begin(), severity.end(), severity.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (severity == "info" || severity == "warning" || severity == "critical") {
    return severity;
  }
  return "error";
}

std::string derive_monitor_error_fingerprint(const std::string& node_id, const std::string& code,
                                             const std::string& message, const std::string& fingerprint) {
  const std::string explicit_fingerprint = trim_copy(fingerprint);
  if (!explicit_fingerprint.empty()) {
    return explicit_fingerprint;
  }
  return node_id + ":" + code + ":" + message;
}

std::uint32_t fnv1a32(const std::string& text) {
  std::uint32_t value = 0x811C9DC5u;
  for (unsigned char ch : text) {
    value ^= static_cast<std::uint32_t>(ch);
    value *= 0x01000193u;
  }
  return value;
}

std::string command_key_for_name(const std::string& name) {
  const std::string raw = trim_copy(name);
  std::string base;
  bool last_was_sep = false;
  for (unsigned char ch : raw) {
    const char lower = static_cast<char>(std::tolower(ch));
    if ((lower >= 'a' && lower <= 'z') || (lower >= '0' && lower <= '9')) {
      base.push_back(lower);
      last_was_sep = false;
      continue;
    }
    if (!last_was_sep) {
      base.push_back('_');
      last_was_sep = true;
    }
  }
  while (!base.empty() && base.front() == '_') base.erase(base.begin());
  while (!base.empty() && base.back() == '_') base.pop_back();
  if (base.empty()) base = "command";
  std::ostringstream out;
  out << base << "_" << std::hex << std::nouppercase << std::setw(8) << std::setfill('0') << fnv1a32(raw);
  return out.str();
}

std::string command_input_state_field(const std::string& name) {
  return "__cmd__." + command_key_for_name(name) + ".in";
}

std::string command_output_state_field(const std::string& name) {
  return "__cmd__." + command_key_for_name(name) + ".out";
}

void stop_runtime_subscriptions(std::vector<std::unique_ptr<RuntimeSubscription>>& subscriptions,
                                const std::string& service_id, const char* action) {
  for (auto& sub : subscriptions) {
    try {
      if (sub) {
        sub->stop();
      }
    } catch (const std::exception& exc) {
      spdlog::warn("{} failed serviceId={}: {}", action, service_id, exc.what());
    } catch (...) {
      spdlog::warn("{} failed serviceId={}: unknown error", action, service_id);
    }
  }
}

json sorted_json_array(const json& arr, const std::function<std::string(const json&)>& key_fn) {
  if (!arr.is_array()) return json::array();
  std::vector<json> items;
  for (const auto& item : arr) {
    items.push_back(item);
  }
  std::sort(items.begin(), items.end(), [&](const json& a, const json& b) { return key_fn(a) < key_fn(b); });
  json out = json::array();
  for (const auto& item : items) {
    out.push_back(item);
  }
  return out;
}

std::string json_string_value(const json& obj, const char* key) {
  if (!obj.is_object() || !obj.contains(key)) return "";
  const auto& value = obj.at(key);
  if (value.is_string()) return value.get<std::string>();
  if (value.is_null()) return "";
  return value.dump();
}

json normalize_spec_payload(const json& payload) {
  if (payload.is_object()) {
    json out = json::object();
    std::vector<std::string> keys;
    for (auto it = payload.begin(); it != payload.end(); ++it) {
      keys.push_back(it.key());
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
      out[key] = normalize_spec_payload(payload.at(key));
    }
    return out;
  }
  if (payload.is_array()) {
    json out = json::array();
    for (const auto& item : payload) {
      out.push_back(normalize_spec_payload(item));
    }
    return out;
  }
  return payload;
}

std::string normalized_named_spec_sort_key(const json& payload) {
  if (!payload.is_object()) return std::string("|") + payload.dump();
  return json_string_value(payload, "name") + "|" + json_string_value(payload, "type") + "|" +
         json_string_value(payload, "access");
}

json normalize_named_specs(const json& specs) {
  if (!specs.is_array()) return json::array();
  json normalized = json::array();
  for (const auto& item : specs) {
    normalized.push_back(normalize_spec_payload(item));
  }
  return sorted_json_array(normalized, normalized_named_spec_sort_key);
}

json normalize_deploy_service_payload(const json& payload) {
  if (!payload.is_object()) return json::object();
  json out = json::object();
  std::vector<std::string> keys;
  for (auto it = payload.begin(); it != payload.end(); ++it) {
    keys.push_back(it.key());
  }
  std::sort(keys.begin(), keys.end());
  for (const auto& key : keys) {
    out[key] = normalize_spec_payload(payload.at(key));
  }
  return out;
}

json normalize_deploy_node_payload(const json& payload) {
  if (!payload.is_object()) return json::object();
  json out = json::object();
  std::vector<std::string> keys;
  for (auto it = payload.begin(); it != payload.end(); ++it) {
    keys.push_back(it.key());
  }
  std::sort(keys.begin(), keys.end());
  for (const auto& key : keys) {
    if (key == "stateValues") continue;
    const auto& value = payload.at(key);
    if ((key == "execInPorts" || key == "execOutPorts") && value.is_array()) {
      std::vector<std::string> ports;
      for (const auto& item : value) {
        ports.push_back(item.is_string() ? item.get<std::string>() : item.dump());
      }
      std::sort(ports.begin(), ports.end());
      out[key] = ports;
      continue;
    }
    if ((key == "dataInPorts" || key == "dataOutPorts" || key == "stateFields") && value.is_array()) {
      out[key] = normalize_named_specs(value);
      continue;
    }
    out[key] = normalize_spec_payload(value);
  }
  return out;
}

json normalize_deploy_edge_payload(const json& payload) {
  if (!payload.is_object()) return json::object();
  json out = json::object();
  std::vector<std::string> keys;
  for (auto it = payload.begin(); it != payload.end(); ++it) {
    if (it.key() == "edgeId") continue;
    keys.push_back(it.key());
  }
  std::sort(keys.begin(), keys.end());
  for (const auto& key : keys) {
    out[key] = normalize_spec_payload(payload.at(key));
  }
  return out;
}

std::string normalized_service_sort_key(const json& payload) {
  return json_string_value(payload, "serviceId") + "|" + json_string_value(payload, "serviceClass");
}

std::string normalized_node_sort_key(const json& payload) {
  return json_string_value(payload, "serviceId") + "|" + json_string_value(payload, "nodeId") + "|" +
         json_string_value(payload, "operatorClass");
}

std::string normalized_edge_sort_key(const json& payload) {
  return json_string_value(payload, "kind") + "|" + json_string_value(payload, "fromServiceId") + "|" +
         json_string_value(payload, "fromOperatorId") + "|" + json_string_value(payload, "fromPort") + "|" +
         json_string_value(payload, "toServiceId") + "|" + json_string_value(payload, "toPort");
}

std::string build_rungraph_deploy_fingerprint(const json& graph_obj) {
  json services = json::array();
  if (graph_obj.is_object() && graph_obj.contains("services") && graph_obj["services"].is_array()) {
    for (const auto& item : graph_obj["services"]) {
      services.push_back(normalize_deploy_service_payload(item));
    }
    services = sorted_json_array(services, normalized_service_sort_key);
  }
  json nodes = json::array();
  if (graph_obj.is_object() && graph_obj.contains("nodes") && graph_obj["nodes"].is_array()) {
    for (const auto& item : graph_obj["nodes"]) {
      nodes.push_back(normalize_deploy_node_payload(item));
    }
    nodes = sorted_json_array(nodes, normalized_node_sort_key);
  }
  json edges = json::array();
  if (graph_obj.is_object() && graph_obj.contains("edges") && graph_obj["edges"].is_array()) {
    for (const auto& item : graph_obj["edges"]) {
      edges.push_back(normalize_deploy_edge_payload(item));
    }
    edges = sorted_json_array(edges, normalized_edge_sort_key);
  }
  json snapshot = json{{"services", services}, {"nodes", nodes}, {"edges", edges}};
  return snapshot.dump(-1, ' ', false, json::error_handler_t::strict);
}

std::string new_control_req_id() {
  return std::to_string(static_cast<long long>(now_ms()));
}

struct ControlEnvelope {
  std::string req_id;
  json raw = json::object();
  json args = json::object();
  json meta = json::object();
};

ControlEnvelope parse_control_envelope(const RuntimeBytes& bytes) {
  ControlEnvelope out;
  if (!bytes.empty() && !decode_json(bytes.data(), bytes.size(), out.raw)) {
    out.raw = json::object();
  }
  if (!out.raw.is_object()) {
    out.raw = json::object();
  }
  if (out.raw.contains("reqId") && out.raw["reqId"].is_string()) {
    out.req_id = out.raw["reqId"].get<std::string>();
  }
  if (out.req_id.empty()) {
    out.req_id = new_control_req_id();
  }
  if (out.raw.contains("args") && out.raw["args"].is_object()) {
    out.args = out.raw["args"];
  }
  if (out.raw.contains("meta") && out.raw["meta"].is_object()) {
    out.meta = out.raw["meta"];
  }
  return out;
}

RuntimeBytes encode_control_response(const std::string& req_id, bool ok, const json& result,
                                     const std::string& err_code, const std::string& err_message) {
  json payload;
  payload["reqId"] = req_id;
  payload["ok"] = ok;
  payload["result"] = ok ? result : json(nullptr);
  if (!ok) {
    std::string code = err_code;
    if (!generated::parse_F8CommandError_code_Enum(code).has_value()) {
      if (code == "INVALID_VALUE" || code == "INVALID_RUNGRAPH" || code == "INVALID_SCHEMA")
        code = "INVALID_ARGS";
      else if (code == "UNKNOWN_FIELD" || code == "NOT_SUPPORTED")
        code = "NOT_FOUND";
      else if (code == "NOT_READY")
        code = "CONFLICT";
      else
        code = "INTERNAL";
    }
    payload["error"] = json{{"code", code}, {"message", err_message}};
  } else {
    payload["error"] = json(nullptr);
  }
  return encode_json(payload);
}

RuntimeBytes encode_ready_payload(const std::string& service_id, bool ready, const std::string& reason,
                                  std::int64_t ts_ms) {
  const std::int64_t ts = ts_ms > 0 ? ts_ms : now_ms();
  json payload = json::object();
  payload["serviceId"] = ensure_token(service_id, "service_id");
  payload["ready"] = ready;
  payload["reason"] = reason;
  payload["ts"] = ts;
  return encode_json(payload);
}

RuntimeBytes encode_node_state_payload(const std::string& service_id, const json& value, const std::string& source,
                                       const json& extra_meta, std::int64_t ts_ms, const std::string& origin) {
  const std::int64_t ts = ts_ms > 0 ? ts_ms : now_ms();
  json payload;
  payload["value"] = value;
  payload["actor"] = ensure_token(service_id, "service_id");
  payload["ts"] = ts;
  if (!source.empty()) {
    payload["source"] = source;
  }
  if (!origin.empty()) {
    payload["origin"] = origin;
  }
  if (extra_meta.is_object()) {
    for (auto it = extra_meta.begin(); it != extra_meta.end(); ++it) {
      const std::string k = it.key();
      if (k == "value" || k == "actor" || k == "ts" || k == "source" || k == "origin") {
        continue;
      }
      payload[k] = it.value();
    }
  }
  return encode_json(payload);
}

RuntimeBytes encode_data_payload(const json& value, std::int64_t ts_ms) {
  const std::int64_t ts = ts_ms > 0 ? ts_ms : now_ms();
  json payload;
  payload["value"] = value;
  payload["ts"] = ts;
  return encode_json(payload);
}

bool hidden_command_state_direction(const std::string& field, std::string& direction) {
  if (field.rfind("__cmd__.", 0) != 0) return false;
  if (field.size() > 3 && field.compare(field.size() - 3, 3, ".in") == 0) {
    direction = "in";
    return true;
  }
  if (field.size() > 4 && field.compare(field.size() - 4, 4, ".out") == 0) {
    direction = "out";
    return true;
  }
  return false;
}

struct ParsedCommandBinding {
  std::string node_id;
  std::string call;
  std::string input_field;
  std::string output_field;
  std::vector<std::string> param_names;
};

std::vector<ParsedCommandBinding> parse_command_bindings_from_spec(const json& spec, const std::string& node_id) {
  std::vector<ParsedCommandBinding> bindings;
  const json* service = nullptr;
  if (spec.is_object() && spec.contains("service") && spec.at("service").is_object()) {
    service = &spec.at("service");
  } else if (spec.is_object()) {
    service = &spec;
  }
  if (service == nullptr || !service->is_object()) return bindings;
  if (!service->contains("commands") || !service->at("commands").is_array()) return bindings;
  for (const auto& command : service->at("commands")) {
    if (!command.is_object()) continue;
    const std::string call = trim_copy(command.value("name", ""));
    if (call.empty()) continue;
    ParsedCommandBinding binding;
    binding.node_id = node_id;
    binding.call = call;
    binding.input_field = command_input_state_field(call);
    binding.output_field = command_output_state_field(call);
    if (command.contains("params") && command.at("params").is_array()) {
      for (const auto& param : command.at("params")) {
        if (!param.is_object()) continue;
        const std::string param_name = trim_copy(param.value("name", ""));
        if (!param_name.empty()) binding.param_names.push_back(param_name);
      }
    }
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

json map_command_args(const json& value, const std::vector<std::string>& param_names) {
  json args = json::object();
  if (param_names.empty()) return args;
  if (value.is_object()) {
    for (const auto& name : param_names) {
      if (value.contains(name)) args[name] = value.at(name);
    }
    return args;
  }
  if (value.is_array()) {
    for (std::size_t i = 0; i < value.size() && i < param_names.size(); ++i) {
      args[param_names[i]] = value.at(i);
    }
    return args;
  }
  args[param_names.front()] = value;
  return args;
}

void prune_timed_values(std::deque<std::pair<std::int64_t, double>>& values, const std::int64_t now_ms,
                        const std::int64_t window_ms) {
  const std::int64_t cutoff = now_ms - std::max<std::int64_t>(1000, window_ms);
  while (!values.empty() && values.front().first < cutoff) {
    values.pop_front();
  }
}

void prune_timed_errors(std::deque<std::int64_t>& values, const std::int64_t now_ms, const std::int64_t window_ms) {
  const std::int64_t cutoff = now_ms - std::max<std::int64_t>(1000, window_ms);
  while (!values.empty() && values.front() < cutoff) {
    values.pop_front();
  }
}

double average_values(const std::deque<std::pair<std::int64_t, double>>& values) {
  if (values.empty()) return 0.0;
  double total = 0.0;
  for (const auto& item : values) {
    total += item.second;
  }
  return total / static_cast<double>(values.size());
}

double percentile95_values(const std::deque<std::pair<std::int64_t, double>>& values) {
  if (values.empty()) return 0.0;
  std::vector<double> ordered;
  ordered.reserve(values.size());
  for (const auto& item : values) {
    ordered.push_back(item.second);
  }
  std::sort(ordered.begin(), ordered.end());
  std::size_t idx = static_cast<std::size_t>(std::llround(static_cast<double>(ordered.size() - 1) * 0.95));
  if (idx >= ordered.size()) idx = ordered.size() - 1;
  return ordered[idx];
}

std::pair<std::int64_t, std::int64_t> sample_process_memory_bytes() {
#if defined(__linux__)
  std::ifstream in("/proc/self/statm");
  if (!in.is_open()) return {0, 0};
  std::uint64_t vms_pages = 0;
  std::uint64_t rss_pages = 0;
  in >> vms_pages >> rss_pages;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) return {0, 0};
  const std::int64_t rss = static_cast<std::int64_t>(rss_pages) * static_cast<std::int64_t>(page_size);
  const std::int64_t vms = static_cast<std::int64_t>(vms_pages) * static_cast<std::int64_t>(page_size);
  return {std::max<std::int64_t>(0, rss), std::max<std::int64_t>(0, vms)};
#else
  return {0, 0};
#endif
}

struct StateEdgeKey {
  std::string service_id;
  std::string node_id;
  std::string field;
  bool operator==(const StateEdgeKey& other) const {
    return service_id == other.service_id && node_id == other.node_id && field == other.field;
  }
};
struct StateEdgeKeyHash {
  std::size_t operator()(const StateEdgeKey& k) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(k.service_id);
    std::size_t h2 = std::hash<std::string>{}(k.node_id);
    std::size_t h3 = std::hash<std::string>{}(k.field);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

void validate_state_edges_or_throw(const f8::cppsdk::generated::F8RuntimeGraph& graph) {
  using namespace f8::cppsdk::generated;

  const auto edges = graph.edges.value_or(std::vector<F8Edge>{});
  std::unordered_map<StateEdgeKey, std::vector<StateEdgeKey>, StateEdgeKeyHash> out;
  std::unordered_map<StateEdgeKey, int, StateEdgeKeyHash> inbound_count;
  std::unordered_map<StateEdgeKey, StateEdgeKey, StateEdgeKeyHash> upstream_by_target;
  std::vector<StateEdgeKey> nodes;

  auto add_node = [&](const StateEdgeKey& k) {
    nodes.push_back(k);
  };

  for (const auto& e : edges) {
    if (e.kind != F8EdgeKindEnum::state) continue;
    const std::string from_sid = e.fromServiceId;
    const std::string to_sid = e.toServiceId;
    const std::string from_op = e.fromOperatorId.value_or("");
    const std::string to_op = e.toOperatorId.value_or("");
    const std::string from_field = e.fromPort;
    const std::string to_field = e.toPort;
    if (from_sid.empty() || to_sid.empty() || from_op.empty() || to_op.empty() || from_field.empty() || to_field.empty()) {
      continue;
    }

    StateEdgeKey from{from_sid, from_op, from_field};
    StateEdgeKey to{to_sid, to_op, to_field};

    auto it_prev = upstream_by_target.find(to);
    if (it_prev != upstream_by_target.end()) {
      const auto& prev = it_prev->second;
      if (!(prev == from)) {
        throw std::runtime_error("multiple upstreams for state field: " + to_sid + "." + to_op + "." + to_field);
      }
    } else {
      upstream_by_target.emplace(to, from);
    }

    out[from].push_back(to);
    inbound_count[to] = inbound_count[to] + 1;
    add_node(from);
    add_node(to);
  }

  if (out.empty()) return;

  std::unordered_map<StateEdgeKey, bool, StateEdgeKeyHash> visiting;
  std::unordered_map<StateEdgeKey, bool, StateEdgeKeyHash> visited;
  std::unordered_map<StateEdgeKey, std::optional<StateEdgeKey>, StateEdgeKeyHash> parent;

  auto fmt = [](const StateEdgeKey& k) { return k.service_id + "." + k.node_id + "." + k.field; };

  std::function<std::optional<std::vector<StateEdgeKey>>(const StateEdgeKey&)> dfs;

  dfs = [&](const StateEdgeKey& n) -> std::optional<std::vector<StateEdgeKey>> {
    visiting[n] = true;
    for (const auto& m : out[n]) {
      if (visited[m]) continue;
      if (visiting[m]) {
        std::vector<StateEdgeKey> cyc;
        cyc.push_back(m);
        cyc.push_back(n);
        auto cur = parent[n];
        while (cur.has_value() && !(cur.value() == m)) {
          cyc.push_back(cur.value());
          cur = parent[cur.value()];
        }
        cyc.push_back(m);
        std::reverse(cyc.begin(), cyc.end());
        return cyc;
      }
      parent[m] = n;
      auto r = dfs(m);
      if (r.has_value()) return r;
    }
    visiting[n] = false;
    visited[n] = true;
    return std::nullopt;
  };

  // Roots first.
  std::vector<StateEdgeKey> start;
  for (const auto& n : nodes) {
    if (inbound_count.find(n) == inbound_count.end()) {
      start.push_back(n);
    }
  }
  for (const auto& n : nodes) {
    if (std::find_if(start.begin(), start.end(), [&](const StateEdgeKey& x) { return x == n; }) == start.end()) {
      start.push_back(n);
    }
  }

  for (const auto& n : start) {
    if (visited[n]) continue;
    parent[n] = std::nullopt;
    auto cyc = dfs(n);
    if (cyc.has_value()) {
      std::string msg = "cyclic state-edge loop detected: ";
      for (std::size_t i = 0; i < cyc->size(); ++i) {
        if (i) msg += " -> ";
        msg += fmt(cyc->at(i));
      }
      throw std::runtime_error(msg);
    }
  }
}

}  // namespace

ServiceBus::ServiceBus(Config cfg) : cfg_(std::move(cfg)), runtime_instance_id_(runtime_instance_id_hex()) {}

ServiceBus::~ServiceBus() {
  stop();
}

void ServiceBus::add_lifecycle_node(LifecycleNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(lifecycle_mu_);
  lifecycle_nodes_.push_back(node);
}

void ServiceBus::add_stateful_node(StatefulNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  stateful_nodes_.push_back(node);
}

void ServiceBus::add_data_node(DataReceivableNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  data_nodes_.push_back(node);
}

void ServiceBus::add_set_state_node(SetStateHandlerNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  set_state_nodes_.push_back(node);
}

void ServiceBus::add_rungraph_node(RungraphHandlerNode* node) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  rungraph_nodes_.push_back(node);
}

void ServiceBus::add_command_node(CommandableNode* node, const json& service_spec) {
  if (node == nullptr) return;
  std::lock_guard<std::mutex> lock(handlers_mu_);
  command_nodes_.push_back(node);
  if (service_spec.is_object()) {
    command_specs_by_node_[node] = service_spec;
  }
}

std::size_t ServiceBus::drain_main_thread(std::size_t max_tasks) {
  return main_thread_.drain(max_tasks);
}

void ServiceBus::post_main_thread(MainThreadQueue::Task task) {
  main_thread_.post(std::move(task));
}

void ServiceBus::handle_data_payload(const std::string& key, const RuntimeBytes& bytes) {
  json payload = json::object();
  if (!decode_json(bytes.data(), bytes.size(), payload)) {
    return;
  }
  if (!payload.is_object()) return;

  json value = payload.contains("value") ? payload["value"] : json();
  const std::int64_t ts_ms = coerce_inbound_ts_ms(payload, 0);

  json meta = payload;
  if (meta.is_object()) {
    meta.erase("value");
    meta["key"] = key;
  } else {
    meta = json::object({{"key", key}});
  }

  const auto value_ptr = std::make_shared<const json>(std::move(value));
  const auto snapshot = std::atomic_load(&data_routes_snapshot_);
  if (!snapshot) return;

  main_thread_.post([this, snapshot, key, value_ptr, ts_ms, meta]() {
    const auto it = snapshot->by_key.find(key);
    if (it == snapshot->by_key.end()) return;
    const auto& routes = it->second;
    if (routes.empty()) return;

    const std::int64_t now = now_ms();
    for (const auto& r : routes) {
      if (!r.buf) continue;
      if (r.timeout_ms > 0 && ts_ms > 0 && (now - ts_ms) > r.timeout_ms) {
        continue;
      }
      auto& buf = *r.buf;
      std::lock_guard<std::mutex> lock(buf.mu);
      std::int64_t dropped = 0;
      if (buf.strategy == EdgeStrategy::kLatest && !buf.queue.empty()) {
        dropped = static_cast<std::int64_t>(buf.queue.size());
      }
      buf.last_seen_value = value_ptr;
      buf.last_seen_ts_ms = ts_ms;
      if (buf.strategy == EdgeStrategy::kLatest) {
        buf.queue.clear();
      }
      buf.queue.emplace_back(value_ptr, ts_ms);
      monitor_record_observed(r.to_port);
      if (dropped > 0) {
        monitor_record_dropped(dropped);
      }
    }

    if (cfg_.data_delivery != DataDeliveryMode::kPush && cfg_.data_delivery != DataDeliveryMode::kBoth) {
      return;
    }

    std::vector<DataReceivableNode*> nodes;
    {
      std::lock_guard<std::mutex> lock(handlers_mu_);
      nodes = data_nodes_;
    }
    if (nodes.empty()) return;

    for (const auto& r : routes) {
      if (r.timeout_ms > 0 && ts_ms > 0 && (now - ts_ms) > r.timeout_ms) {
        continue;
      }
      json m = meta;
      m["fromServiceId"] = r.from_service_id;
      m["fromNodeId"] = r.from_node_id;
      m["fromPort"] = r.from_port;
      for (auto* n : nodes) {
        if (!n) continue;
        try {
          n->on_data(r.to_node_id, r.to_port, *value_ptr, ts_ms, m);
        } catch (const std::exception& exc) {
          monitor_record_error("DATA_CALLBACK_FAILED", exc.what(), ts_ms);
        } catch (...) {
          monitor_record_error("DATA_CALLBACK_FAILED", "on_data threw unknown exception", ts_ms);
        }
      }
    }
  });
}

void ServiceBus::handle_peer_state_payload(const std::string& peer, const std::string& key, const RuntimeBytes& bytes) {
  const auto state_path = zenoh_key_to_state_path(key);
  if (!state_path.has_value()) return;
  constexpr const char* kPrefix = "nodes.";
  constexpr const char* kStateMarker = ".state.";
  if (state_path->rfind(kPrefix, 0) != 0) return;
  const std::size_t marker = state_path->find(kStateMarker);
  if (marker == std::string::npos) return;

  const std::size_t node_begin = std::strlen(kPrefix);
  const std::size_t node_end = marker;
  if (node_end <= node_begin) return;
  const std::string remote_node_id = state_path->substr(node_begin, node_end - node_begin);

  const std::size_t field_begin = marker + std::strlen(kStateMarker);
  if (field_begin >= state_path->size()) return;
  const std::string remote_field = state_path->substr(field_begin);

  std::vector<_NodeFieldKey> targets;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = cross_state_in_.find(_RemoteStateKey{peer, remote_node_id, remote_field});
    if (it == cross_state_in_.end()) return;
    targets = it->second;
  }
  if (targets.empty()) return;

  nlohmann::json payload = nlohmann::json::object();
  if (!decode_json(bytes.data(), bytes.size(), payload)) {
    return;
  }
  if (!payload.is_object()) return;

  const nlohmann::json value = payload.contains("value") ? payload["value"] : nlohmann::json();
  const std::int64_t ts_ms = coerce_inbound_ts_ms(payload, now_ms());
  nlohmann::json meta = payload;
  if (meta.is_object()) {
    meta.erase("value");
  } else {
    meta = nlohmann::json::object();
  }
  meta["peerServiceId"] = peer;
  meta["remoteKey"] = key;
  meta["fromNodeId"] = remote_node_id;
  meta["fromField"] = remote_field;

  if (state_debug_enabled()) {
    std::string v_s;
    try {
      v_s = value.dump();
    } catch (const std::exception& exc) {
      v_s = std::string("<json_dump_failed: ") + exc.what() + ">";
    } catch (...) {
      v_s = "<json_dump_failed: unknown error>";
    }
    if (v_s.size() > 160) v_s = v_s.substr(0, 157) + "...";
    spdlog::info("state_debug[{}] cross_state_watch peer={} key={} ts={} targets={} value={}", cfg_.service_id,
                 peer, key, ts_ms, targets.size(), v_s);
  }

  for (const auto& t : targets) {
    publish_state_local(t.node_id, t.field, value, ts_ms, "state_edge_cross", meta, "external", true, true);
  }
}

void ServiceBus::apply_data_routes_from_rungraph(const json& graph_obj) {
  auto new_routes = parse_cross_service_data_routes(graph_obj, cfg_.service_id);

  std::vector<std::unique_ptr<RuntimeSubscription>> removed_subscriptions;

  // Build new routing snapshot + input buffers.
  auto next_snapshot = std::make_shared<_DataRoutingSnapshot>();
  auto next_inputs = std::unordered_map<_NodePortKey, std::shared_ptr<_InputBuffer>, _NodePortKeyHash>();
  auto next_stream_keys = std::unordered_map<_NodePortKey, std::string, _NodePortKeyHash>();

  for (const auto& route_entry : new_routes) {
    const std::string& key = route_entry.first;
    std::vector<_RouteRuntime> vec;
    vec.reserve(route_entry.second.size());
    for (const auto& r : route_entry.second) {
      if (r.stream_payload) {
        next_stream_keys[{r.to_node_id, r.to_port}] = key;
        continue;
      }
      _NodePortKey key{r.to_node_id, r.to_port};
      auto it = next_inputs.find(key);
      if (it == next_inputs.end()) {
        it = next_inputs.emplace(key, std::make_shared<_InputBuffer>()).first;
      }
      auto& buf = *it->second;
      if (r.strategy == EdgeStrategy::kQueue) {
        buf.strategy = EdgeStrategy::kQueue;
      }
      if (r.timeout_ms > 0) {
        if (buf.timeout_ms <= 0) {
          buf.timeout_ms = r.timeout_ms;
        } else {
          buf.timeout_ms = std::min<std::int64_t>(buf.timeout_ms, r.timeout_ms);
        }
      }

      _RouteRuntime rr;
      rr.to_node_id = r.to_node_id;
      rr.to_port = r.to_port;
      rr.from_service_id = r.from_service_id;
      rr.from_node_id = r.from_node_id;
      rr.from_port = r.from_port;
      rr.strategy = r.strategy;
      rr.timeout_ms = r.timeout_ms;
      rr.buf = it->second;
      vec.push_back(std::move(rr));
    }
    if (!vec.empty()) {
      next_snapshot->by_key.emplace(key, std::move(vec));
    }
  }

  {
    std::lock_guard<std::mutex> lock(data_mu_);

    // Unsubscribe removed keys. Stop them after releasing data_mu_ so callbacks
    // cannot deadlock while trying to publish into the same service bus.
    for (auto it = runtime_data_subs_.begin(); it != runtime_data_subs_.end();) {
      if (next_snapshot->by_key.find(it->first) != next_snapshot->by_key.end()) {
        ++it;
        continue;
      }
      removed_subscriptions.push_back(std::move(it->second));
      it = runtime_data_subs_.erase(it);
    }

    data_inputs_ = std::move(next_inputs);
    data_input_stream_keys_ = std::move(next_stream_keys);
    std::shared_ptr<const _DataRoutingSnapshot> next_snapshot_const = next_snapshot;
    std::atomic_store(&data_routes_snapshot_, std::move(next_snapshot_const));
  }

  stop_runtime_subscriptions(removed_subscriptions, cfg_.service_id, "runtime data unsubscribe");

  // Subscribe new keys.
  for (const auto& route_entry : next_snapshot->by_key) {
    const std::string& key = route_entry.first;
    {
      std::lock_guard<std::mutex> lock(data_mu_);
      if (runtime_data_subs_.find(key) != runtime_data_subs_.end()) {
        continue;
      }
    }
    if (!runtime_transport_) {
      spdlog::warn("runtime data subscription skipped without transport serviceId={} key={}", cfg_.service_id,
                   key);
      continue;
    }
    auto sub = runtime_transport_->subscribe(key, [this](const RuntimeMessage& msg) {
      handle_data_payload(msg.key, msg.payload);
    });
    if (sub && sub->valid()) {
      std::unique_ptr<RuntimeSubscription> duplicate_sub;
      {
        std::lock_guard<std::mutex> lock(data_mu_);
        const auto [it, inserted] = runtime_data_subs_.try_emplace(key, std::move(sub));
        if (!inserted) {
          duplicate_sub = std::move(sub);
        }
        (void)it;
      }
      if (duplicate_sub) {
        std::vector<std::unique_ptr<RuntimeSubscription>> duplicate_subscriptions;
        duplicate_subscriptions.push_back(std::move(duplicate_sub));
        stop_runtime_subscriptions(duplicate_subscriptions, cfg_.service_id, "runtime data duplicate subscription stop");
      }
    } else {
      spdlog::warn("runtime data subscription failed serviceId={} key={}", cfg_.service_id, key);
    }
  }
}

bool ServiceBus::start() {
  stop();

  cfg_.service_id = ensure_token(cfg_.service_id, "service_id");
  cfg_.runtime_instance_id = runtime_instance_id_;
  if (cfg_.service_name.empty()) {
    cfg_.service_name = cfg_.service_id;
  }
  cfg_.apply_runtime_backend(cfg_.runtime_backend_config());
  terminate_.store(false, std::memory_order_release);
  ready_.store(false, std::memory_order_release);

  if (cfg_.bus_backend == BusBackend::kZenoh) {
    return start_zenoh_backend();
  }

  spdlog::error("service_bus backend={} is not implemented for C++ serviceId={}",
                bus_backend_to_string(cfg_.bus_backend), cfg_.service_id);
  return false;
}

bool ServiceBus::start_zenoh_backend() {
  runtime_transport_ = std::make_unique<ZenohTransport>();
  if (!runtime_transport_->connect(cfg_.runtime_backend_config(), cfg_.service_id)) {
    runtime_transport_.reset();
    return false;
  }
  start_rungraph_apply_worker();

  if (!start_runtime_control_endpoints()) {
    stop_runtime_control_endpoints();
    stop_rungraph_apply_worker();
    runtime_transport_->close();
    runtime_transport_.reset();
    return false;
  }

  load_active_from_retained();

  (void)runtime_set_ready(false, "starting");
  ready_.store(false, std::memory_order_release);

  (void)runtime_set_node_state(cfg_.service_id, "svcId", cfg_.service_id, "system", json{{"builtin", true}}, 0,
                               "system");
  (void)runtime_set_node_state(cfg_.service_id, "active", active_.load(std::memory_order_acquire), "system",
                               json{{"builtin", true}, {"bootstrap", true}}, 0, "runtime");

  (void)runtime_set_ready(true, "start");
  ready_.store(true, std::memory_order_release);
  start_monitor_thread();
  spdlog::info("service_bus started serviceId={} backend={}", cfg_.service_id, bus_backend_to_string(cfg_.bus_backend));
  return true;
}

bool ServiceBus::start_runtime_control_endpoints() {
  if (!runtime_transport_) {
    spdlog::error("runtime control endpoints require an active runtime transport serviceId={}", cfg_.service_id);
    return false;
  }

  struct EndpointRegistration {
    std::string endpoint;
    std::string key;
  };

  const std::vector<EndpointRegistration> registrations = {
      {"activate", svc_endpoint_key(cfg_.service_id, "activate")},
      {"deactivate", svc_endpoint_key(cfg_.service_id, "deactivate")},
      {"set_active", svc_endpoint_key(cfg_.service_id, "set_active")},
      {"status", svc_endpoint_key(cfg_.service_id, "status")},
      {"terminate", svc_endpoint_key(cfg_.service_id, "terminate")},
      {"quit", svc_endpoint_key(cfg_.service_id, "quit")},
      {"cmd", cmd_channel_key(cfg_.service_id)},
      {"set_state", svc_endpoint_key(cfg_.service_id, "set_state")},
      {"set_rungraph", svc_endpoint_key(cfg_.service_id, "set_rungraph")},
  };

  for (const EndpointRegistration& registration : registrations) {
    auto handle = runtime_transport_->serve(
        registration.key,
        [this, endpoint = registration.endpoint](const RuntimeMessage& msg) {
          return handle_runtime_control_request(endpoint, msg);
        });
    if (!handle || !handle->valid()) {
      spdlog::error("runtime control endpoint registration failed serviceId={} endpoint={} key={}",
                    cfg_.service_id, registration.endpoint, registration.key);
      stop_runtime_control_endpoints();
      return false;
    }
    runtime_control_endpoints_.push_back(std::move(handle));
  }
  return true;
}

void ServiceBus::stop_runtime_control_endpoints() {
  for (auto& handle : runtime_control_endpoints_) {
    if (!handle) {
      continue;
    }
    try {
      handle->stop();
    } catch (const std::exception& exc) {
      spdlog::warn("runtime control endpoint stop failed serviceId={}: {}", cfg_.service_id, exc.what());
    } catch (...) {
      spdlog::warn("runtime control endpoint stop failed serviceId={}: unknown error", cfg_.service_id);
    }
  }
  runtime_control_endpoints_.clear();
}

RuntimeBytes ServiceBus::handle_runtime_control_request(const std::string& endpoint, const RuntimeMessage& msg) {
  const auto env = parse_control_envelope(msg.payload);
  std::string err_code;
  std::string err_msg;
  json result = json::object();

  auto ok_response = [&](const json& out) { return encode_control_response(env.req_id, true, out, "", ""); };
  auto error_response = [&](const std::string& code, const std::string& message) {
    return encode_control_response(env.req_id, false, json(nullptr), code, message);
  };

  try {
    if (endpoint == "activate") {
      on_activate(env.meta);
      return ok_response(json{{"active", true}});
    }
    if (endpoint == "deactivate") {
      on_deactivate(env.meta);
      return ok_response(json{{"active", false}});
    }
    if (endpoint == "set_active") {
      const json* src = nullptr;
      if (env.args.contains("active")) {
        src = &env.args;
      } else if (env.raw.contains("active")) {
        src = &env.raw;
      } else {
        return error_response("INVALID_ARGS", "missing active");
      }

      f8::cppsdk::generated::F8SetActiveArgs req;
      f8::cppsdk::generated::ParseError perr;
      if (!f8::cppsdk::generated::parse_F8SetActiveArgs(*src, req, perr)) {
        return error_response("INVALID_ARGS", perr.message.empty() ? "invalid request" : perr.message);
      }
      on_set_active(req.active, env.meta);
      return ok_response(json{{"active", req.active}});
    }
    if (endpoint == "status") {
      const _RungraphMetadata rungraph_metadata = rungraph_metadata_snapshot();
      return ok_response(json{{"serviceId", cfg_.service_id},
                              {"serviceClass", cfg_.service_class},
                              {"runtimeInstanceId", runtime_instance_id_},
                              {"active", is_active()},
                              {"rungraphGraphId", rungraph_metadata.graph_id},
                              {"rungraphRevision", rungraph_metadata.revision},
                              {"rungraphFingerprint", rungraph_metadata.fingerprint}});
    }
    if (endpoint == "terminate" || endpoint == "quit") {
      spdlog::info("{} requested serviceId={}", endpoint, cfg_.service_id);
      json out;
      const bool ok = on_command("terminate", env.args, env.meta, out, err_code, err_msg);
      if (!ok) {
        return error_response(err_code, err_msg);
      }
      return ok_response(json{{"terminating", true}});
    }
    if (endpoint == "set_state") {
      const json* src = nullptr;
      if (env.args.contains("nodeId") || env.args.contains("field") || env.args.contains("value")) {
        src = &env.args;
      } else if (env.raw.contains("nodeId") || env.raw.contains("field") || env.raw.contains("value")) {
        src = &env.raw;
      } else {
        return error_response("INVALID_ARGS", "missing nodeId/field/value");
      }

      f8::cppsdk::generated::F8SetStateArgs req;
      f8::cppsdk::generated::ParseError perr;
      if (!f8::cppsdk::generated::parse_F8SetStateArgs(*src, req, perr)) {
        return error_response("INVALID_ARGS", perr.message.empty() ? "invalid request" : perr.message);
      }

      const bool ok = on_set_state(req.nodeId, req.field, req.value, env.meta, err_code, err_msg);
      if (!ok) {
        return error_response(err_code, err_msg);
      }
      return ok_response(json{{"nodeId", req.nodeId}, {"field", req.field}});
    }
    if (endpoint == "set_rungraph") {
      json graph_obj;
      f8::cppsdk::generated::ParseError perr;
      if (env.args.contains("graph") && env.args["graph"].is_object()) {
        f8::cppsdk::generated::F8SetRungraphArgs req;
        if (!f8::cppsdk::generated::parse_F8SetRungraphArgs(env.args, req, perr)) {
          return error_response("INVALID_ARGS", perr.message.empty() ? "invalid request" : perr.message);
        }
        graph_obj = env.args["graph"];
      } else if (env.raw.contains("graph") && env.raw["graph"].is_object()) {
        f8::cppsdk::generated::F8SetRungraphArgs req;
        if (!f8::cppsdk::generated::parse_F8SetRungraphArgs(env.raw, req, perr)) {
          return error_response("INVALID_ARGS", perr.message.empty() ? "invalid request" : perr.message);
        }
        graph_obj = env.raw["graph"];
      } else if (env.raw.is_object() && env.raw.contains("nodes") && env.raw.contains("edges")) {
        f8::cppsdk::generated::F8RuntimeGraph req;
        if (!f8::cppsdk::generated::parse_F8RuntimeGraph(env.raw, req, perr)) {
          return error_response("INVALID_ARGS", perr.message.empty() ? "invalid request" : perr.message);
        }
        graph_obj = env.raw;
      } else {
        return error_response("INVALID_ARGS", "missing graph");
      }

      const bool ok = submit_rungraph(graph_obj, env.meta, env.req_id, err_code, err_msg);
      if (!ok) {
        return error_response(err_code, err_msg);
      }
      return ok_response(json{{"graphId", graph_obj.value("graphId", "")}});
    }
    if (endpoint == "cmd") {
      f8::cppsdk::generated::F8CommandInvokeRequest req;
      f8::cppsdk::generated::ParseError perr;
      if (!f8::cppsdk::generated::parse_F8CommandInvokeRequest(env.raw, req, perr)) {
        return error_response("INVALID_ARGS", perr.message.empty() ? "invalid request" : perr.message);
      }
      json out;
      const bool ok = on_command(req.call, req.args, req.meta, out, err_code, err_msg);
      if (!ok) {
        return error_response(err_code, err_msg);
      }
      return ok_response(out);
    }
  } catch (const std::exception& exc) {
    spdlog::error("runtime control endpoint failed serviceId={} endpoint={}: {}", cfg_.service_id, endpoint, exc.what());
    return error_response("INTERNAL", exc.what());
  } catch (...) {
    spdlog::error("runtime control endpoint failed serviceId={} endpoint={}: unknown error", cfg_.service_id, endpoint);
    return error_response("INTERNAL", "unknown error");
  }

  return error_response("NOT_FOUND", "unknown endpoint");
}

bool ServiceBus::runtime_publish_data(const std::string& from_node_id, const std::string& port_id, const json& value,
                                      std::int64_t ts_ms) {
  const auto key = data_key(cfg_.service_id, from_node_id, port_id);
  const auto bytes = encode_data_payload(value, ts_ms);
  if (runtime_transport_) {
    return runtime_transport_->publish(key, bytes);
  }
  return false;
}

bool ServiceBus::runtime_retained_put(const std::string& key, const RuntimeBytes& bytes) {
  if (runtime_transport_) {
    return runtime_transport_->retained_put(key, bytes);
  }
  return false;
}

std::optional<RuntimeBytes> ServiceBus::runtime_retained_get(const std::string& key) {
  if (runtime_transport_) {
    return runtime_transport_->retained_get(key);
  }
  return std::nullopt;
}

bool ServiceBus::runtime_set_ready(bool ready, const std::string& reason, std::int64_t ts_ms) {
  const auto raw = encode_ready_payload(cfg_.service_id, ready, reason, ts_ms);
  return runtime_retained_put(ready_key(cfg_.service_id), raw);
}

bool ServiceBus::runtime_set_node_state(const std::string& node_id, const std::string& field, const json& value,
                                        const std::string& source, const json& extra_meta, std::int64_t ts_ms,
                                        const std::string& origin) {
  const auto raw = encode_node_state_payload(cfg_.service_id, value, source, extra_meta, ts_ms, origin);
  return runtime_retained_put(zenoh_state_key(cfg_.service_id, node_id, field), raw);
}

void ServiceBus::stop() {
  ready_.store(false, std::memory_order_release);
  stop_monitor_thread();
  if (runtime_transport_ && !cfg_.service_id.empty()) {
    (void)runtime_set_ready(false, "stop");
  }
  stop_runtime_control_endpoints();
  stop_rungraph_apply_worker();

  std::vector<std::unique_ptr<RuntimeSubscription>> peer_state_subs_to_stop;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    for (auto& sub_entry : peer_state_subs_by_service_id_) {
      peer_state_subs_to_stop.push_back(std::move(sub_entry.second));
    }
    peer_state_subs_by_service_id_.clear();
    cross_state_in_.clear();
    cross_state_targets_.clear();
  }
  stop_runtime_subscriptions(peer_state_subs_to_stop, cfg_.service_id, "peer state subscription stop");

  std::vector<std::unique_ptr<RuntimeSubscription>> runtime_data_subs_to_stop;
  {
    std::lock_guard<std::mutex> lock(data_mu_);
    for (auto& sub_entry : runtime_data_subs_) {
      runtime_data_subs_to_stop.push_back(std::move(sub_entry.second));
    }
    runtime_data_subs_.clear();
    data_inputs_.clear();
    data_input_stream_keys_.clear();
    std::atomic_store(&data_routes_snapshot_, std::shared_ptr<const _DataRoutingSnapshot>{});
  }
  stop_runtime_subscriptions(runtime_data_subs_to_stop, cfg_.service_id, "runtime data subscription stop");
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_cache_.clear();
    state_access_.clear();
    intra_state_out_.clear();
    cross_state_in_.clear();
    cross_state_targets_.clear();
    has_rungraph_ = false;
  }
  try {
    main_thread_.clear();
  } catch (const std::exception& exc) {
    spdlog::warn("main-thread queue clear failed serviceId={}: {}", cfg_.service_id, exc.what());
  } catch (...) {
    spdlog::warn("main-thread queue clear failed serviceId={}: unknown error", cfg_.service_id);
  }
  if (runtime_transport_) {
    runtime_transport_->close();
    runtime_transport_.reset();
  }
}

void ServiceBus::wait_terminate() {
  std::unique_lock<std::mutex> lock(term_mu_);
  term_cv_.wait(lock, [this]() { return terminate_.load(std::memory_order_acquire); });
}

void ServiceBus::set_active_local(bool active, const json& meta, const std::string& source) {
  active_.store(active, std::memory_order_release);

  // Persist `nodes.<serviceId>.state.active` (mirror pysdk).
  const json extra = meta.is_object() ? meta : json::object();
  (void)runtime_set_node_state(cfg_.service_id, "active", active, source, extra, 0, "runtime");

  std::vector<LifecycleNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    nodes = lifecycle_nodes_;
  }
  for (const auto& n : nodes) {
    try {
      if (n) n->on_lifecycle(active, meta);
    } catch (const std::exception& exc) {
      spdlog::warn("lifecycle callback failed serviceId={}: {}", cfg_.service_id, exc.what());
    } catch (...) {
      spdlog::warn("lifecycle callback failed serviceId={}: unknown error", cfg_.service_id);
      continue;
    }
  }
}

void ServiceBus::start_monitor_thread() {
  stop_monitor_thread();
  if (!cfg_.monitor_enabled) return;
  monitor_started_ts_ms_ = now_ms();
  {
    std::lock_guard<std::mutex> wake_lock(monitor_wake_mu_);
    monitor_publish_requested_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(monitor_mu_);
    monitor_observed_ = 0;
    monitor_processed_ = 0;
    monitor_dropped_ = 0;
    monitor_wait_ms_.clear();
    monitor_process_ms_.clear();
    monitor_latency_ms_.clear();
    monitor_error_ts_ms_.clear();
    monitor_last_error_code_.clear();
    monitor_last_error_message_.clear();
    monitor_last_error_ts_ms_.reset();
  }
  monitor_running_.store(true, std::memory_order_release);
  monitor_thread_ = std::thread([this]() { monitor_loop(); });
}

void ServiceBus::stop_monitor_thread() {
  monitor_running_.store(false, std::memory_order_release);
  monitor_wake_cv_.notify_all();
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }
}

void ServiceBus::request_monitor_publish_once() {
  if (!cfg_.monitor_enabled) return;
  if (!monitor_running_.load(std::memory_order_acquire)) return;
  {
    std::lock_guard<std::mutex> wake_lock(monitor_wake_mu_);
    monitor_publish_requested_ = true;
  }
  monitor_wake_cv_.notify_all();
}

void ServiceBus::monitor_record_observed(const std::string& port) {
  if (!cfg_.monitor_enabled) return;
  if (port == "monitor") return;
  std::lock_guard<std::mutex> lock(monitor_mu_);
  ++monitor_observed_;
}

void ServiceBus::monitor_record_processed(const std::string& port, const std::int64_t emit_ts_ms,
                                          const std::int64_t now_ts_ms) {
  (void)emit_ts_ms;
  (void)now_ts_ms;
  if (!cfg_.monitor_enabled) return;
  if (port == "monitor") return;
  std::lock_guard<std::mutex> lock(monitor_mu_);
  ++monitor_processed_;
}

void ServiceBus::monitor_record_timing(const std::string& port, double process_ms, double latency_ms,
                                       std::int64_t ts_ms) {
  if (!cfg_.monitor_enabled) return;
  if (port == "monitor") return;
  if (ts_ms <= 0) ts_ms = now_ms();
  std::lock_guard<std::mutex> lock(monitor_mu_);
  if (process_ms >= 0.0 && std::isfinite(process_ms)) {
    monitor_process_ms_.push_back({ts_ms, process_ms});
  }
  if (latency_ms >= 0.0 && std::isfinite(latency_ms)) {
    monitor_latency_ms_.push_back({ts_ms, latency_ms});
  }
}

void ServiceBus::monitor_record_wait_ms(const double wait_ms) {
  if (!cfg_.monitor_enabled) return;
  if (wait_ms < 0.0) return;
  const std::int64_t ts = now_ms();
  std::lock_guard<std::mutex> lock(monitor_mu_);
  monitor_wait_ms_.push_back({ts, wait_ms});
}

void ServiceBus::monitor_record_dropped(const std::int64_t dropped_count) {
  if (!cfg_.monitor_enabled) return;
  if (dropped_count <= 0) return;
  std::lock_guard<std::mutex> lock(monitor_mu_);
  monitor_dropped_ += static_cast<std::uint64_t>(dropped_count);
}

void ServiceBus::report_error(const std::string& node_id, const std::string& code, const std::string& message,
                              const std::string& severity, const std::string& fingerprint, std::int64_t ts_ms) {
  if (!cfg_.monitor_enabled) return;
  if (ts_ms <= 0) ts_ms = now_ms();
  std::string node_id_s = trim_copy(node_id);
  if (node_id_s.empty()) node_id_s = cfg_.service_id;
  std::string code_s = trim_copy(code);
  if (code_s.empty()) code_s = "ERROR";
  const std::string severity_s = normalize_monitor_error_severity(severity);
  const std::string fingerprint_s = derive_monitor_error_fingerprint(node_id_s, code_s, message, fingerprint);

  std::lock_guard<std::mutex> lock(monitor_mu_);
  if (fingerprint_s == monitor_last_error_fingerprint_) {
    monitor_last_error_repeat_count_ = std::max<std::int64_t>(1, monitor_last_error_repeat_count_) + 1;
  } else {
    monitor_last_error_repeat_count_ = 1;
  }
  monitor_last_error_node_id_ = node_id_s;
  monitor_last_error_code_ = code_s;
  monitor_last_error_message_ = message;
  monitor_last_error_severity_ = severity_s;
  monitor_last_error_fingerprint_ = fingerprint_s;
  monitor_last_error_ts_ms_ = ts_ms;
  monitor_current_error_node_id_ = node_id_s;
  monitor_current_error_code_ = code_s;
  monitor_current_error_message_ = message;
  monitor_current_error_severity_ = severity_s;
  monitor_current_error_fingerprint_ = fingerprint_s;
  monitor_current_error_ts_ms_ = ts_ms;
  monitor_error_ts_ms_.push_back(ts_ms);
  request_monitor_publish_once();
}

void ServiceBus::clear_error(const std::string& node_id, const std::string& fingerprint, std::int64_t ts_ms) {
  if (!cfg_.monitor_enabled) return;
  std::string node_id_s = trim_copy(node_id);
  if (node_id_s.empty()) node_id_s = cfg_.service_id;
  const std::string fingerprint_s = trim_copy(fingerprint);
  std::lock_guard<std::mutex> lock(monitor_mu_);
  if (!monitor_current_error_node_id_.empty() && monitor_current_error_node_id_ != node_id_s) {
    return;
  }
  if (!fingerprint_s.empty() && !monitor_current_error_fingerprint_.empty() &&
      monitor_current_error_fingerprint_ != fingerprint_s) {
    return;
  }
  monitor_current_error_node_id_.clear();
  monitor_current_error_code_.clear();
  monitor_current_error_message_.clear();
  monitor_current_error_severity_.clear();
  monitor_current_error_fingerprint_.clear();
  monitor_current_error_ts_ms_.reset();
  (void)ts_ms;
  request_monitor_publish_once();
}

void ServiceBus::record_monitor_timing(const std::string& port, double process_ms, double latency_ms,
                                       std::int64_t ts_ms) {
  monitor_record_timing(port, process_ms, latency_ms, ts_ms);
}

void ServiceBus::record_monitor_processed(const std::string& port, std::int64_t ts_ms) {
  const std::int64_t now_ts = ts_ms > 0 ? ts_ms : now_ms();
  monitor_record_processed(port, 0, now_ts);
}

void ServiceBus::monitor_record_error(const std::string& code, const std::string& message, std::int64_t ts_ms) {
  report_error(cfg_.service_id, code, message, "error", "", ts_ms);
}

std::size_t ServiceBus::monitor_queue_depth() const {
  std::size_t depth = 0;
  std::lock_guard<std::mutex> lock(data_mu_);
  for (const auto& input_entry : data_inputs_) {
    const std::shared_ptr<_InputBuffer>& buf_ptr = input_entry.second;
    if (!buf_ptr) continue;
    std::lock_guard<std::mutex> buf_lock(buf_ptr->mu);
    depth += buf_ptr->queue.size();
  }
  return depth;
}

void ServiceBus::monitor_loop() {
  using clock = std::chrono::steady_clock;
  const std::int64_t interval_ms = std::max<std::int64_t>(200, cfg_.monitor_interval_ms);
  const std::int64_t window_ms = std::max<std::int64_t>(1000, cfg_.monitor_window_ms);
  std::uint32_t cpu_count_raw = std::thread::hardware_concurrency();
  if (cpu_count_raw == 0) {
    cpu_count_raw = 1;
  }
  const double cpu_count = static_cast<double>(cpu_count_raw);
  auto last_wall = clock::now();
  std::clock_t last_cpu = std::clock();

  while (monitor_running_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> wake_lock(monitor_wake_mu_);
      monitor_wake_cv_.wait_for(wake_lock, std::chrono::milliseconds(interval_ms), [this]() {
        return !monitor_running_.load(std::memory_order_acquire) || monitor_publish_requested_;
      });
      monitor_publish_requested_ = false;
    }
    if (!monitor_running_.load(std::memory_order_acquire)) {
      break;
    }

    const std::int64_t ts = now_ms();
    const auto current_wall = clock::now();
    const std::clock_t current_cpu = std::clock();
    const double wall_s = std::chrono::duration<double>(current_wall - last_wall).count();
    const double cpu_s = static_cast<double>(current_cpu - last_cpu) / static_cast<double>(CLOCKS_PER_SEC);
    last_wall = current_wall;
    last_cpu = current_cpu;

    double process_percent = 0.0;
    if (wall_s > 0.0) {
      process_percent = (cpu_s / wall_s) * 100.0 / cpu_count;
      if (process_percent < 0.0) process_percent = 0.0;
    }

    const auto memory = sample_process_memory_bytes();
    std::uint64_t observed = 0;
    std::uint64_t processed = 0;
    std::uint64_t dropped = 0;
    double process_avg = 0.0;
    double process_p95 = 0.0;
    double wait_avg = 0.0;
    double wait_p95 = 0.0;
    double latency_avg = 0.0;
    double latency_p95 = 0.0;
    std::size_t error_count_window = 0;
    std::string last_error_node_id;
    std::string last_error_code;
    std::string last_error_message;
    std::string last_error_severity = "error";
    std::string last_error_fingerprint;
    std::int64_t last_error_repeat_count = 0;
    std::optional<std::int64_t> last_error_ts_ms;
    std::string current_error_node_id;
    std::string current_error_code;
    std::string current_error_message;
    std::string current_error_severity;
    std::optional<std::int64_t> current_error_ts_ms;

    {
      std::lock_guard<std::mutex> lock(monitor_mu_);
      prune_timed_values(monitor_wait_ms_, ts, window_ms);
      prune_timed_values(monitor_process_ms_, ts, window_ms);
      prune_timed_values(monitor_latency_ms_, ts, window_ms);
      prune_timed_errors(monitor_error_ts_ms_, ts, window_ms);
      observed = monitor_observed_;
      processed = monitor_processed_;
      dropped = monitor_dropped_;
      process_avg = average_values(monitor_process_ms_);
      process_p95 = percentile95_values(monitor_process_ms_);
      wait_avg = average_values(monitor_wait_ms_);
      wait_p95 = percentile95_values(monitor_wait_ms_);
      latency_avg = average_values(monitor_latency_ms_);
      latency_p95 = percentile95_values(monitor_latency_ms_);
      error_count_window = monitor_error_ts_ms_.size();
      last_error_node_id = monitor_last_error_node_id_;
      last_error_code = monitor_last_error_code_;
      last_error_message = monitor_last_error_message_;
      last_error_severity = monitor_last_error_severity_.empty() ? "error" : monitor_last_error_severity_;
      last_error_fingerprint = monitor_last_error_fingerprint_;
      last_error_repeat_count = monitor_last_error_repeat_count_;
      last_error_ts_ms = monitor_last_error_ts_ms_;
      current_error_node_id = monitor_current_error_node_id_;
      current_error_code = monitor_current_error_code_;
      current_error_message = monitor_current_error_message_;
      current_error_severity = monitor_current_error_severity_;
      current_error_ts_ms = monitor_current_error_ts_ms_;
    }

    const json gpu = json{
        {"vendor", cfg_.monitor_gpu_enabled ? "nvidia" : ""},
        {"deviceIndex", nullptr},
        {"utilPercent", nullptr},
        {"memoryUsedBytes", nullptr},
        {"memoryTotalBytes", nullptr},
        {"available", false},
    };
    const json snapshot = json{
        {"schemaVersion", "f8monitor/1"},
        {"serviceId", cfg_.service_id},
        {"serviceClass", cfg_.service_class},
        {"nodeId", cfg_.service_id},
        {"tsMs", ts},
        {"alive", true},
        {"ready", ready_.load(std::memory_order_acquire)},
        {"active", active_.load(std::memory_order_acquire)},
        {"uptimeMs", std::max<std::int64_t>(0, ts - monitor_started_ts_ms_)},
        {"cpu", json{{"processPercent", process_percent}, {"systemPercent", 0.0}}},
        {"memory", json{{"rssBytes", memory.first}, {"vmsBytes", memory.second}}},
        {"gpu", gpu},
        {"frame", json{{"observed", observed},
                       {"processed", processed},
                       {"dropped", dropped},
                       {"localOnlyEmits", 0},
                       {"routedCrossEmits", 0},
                       {"suppressedCrossPublishes", 0},
                       {"callbackDeliveries", 0},
                       {"bufferPullDeliveries", 0}}},
        {"timing", json{{"processMsAvg", process_avg},
                        {"processMsP95", process_p95},
                        {"waitMsAvg", wait_avg},
                        {"waitMsP95", wait_p95},
                        {"latencyMsAvg", latency_avg},
                        {"latencyMsP95", latency_p95}}},
        {"queue", json{{"depth", monitor_queue_depth()}}},
        {"error", json{{"countWindow", error_count_window},
                       {"lastNodeId", last_error_node_id},
                       {"lastCode", last_error_code},
                       {"lastMessage", last_error_message},
                       {"lastSeverity", last_error_severity},
                       {"lastFingerprint", last_error_fingerprint},
                       {"lastRepeatCount", last_error_repeat_count},
                       {"lastTsMs", last_error_ts_ms.has_value() ? json(last_error_ts_ms.value()) : json(nullptr)},
                       {"currentNodeId", current_error_node_id},
                       {"currentCode", current_error_code},
                       {"currentMessage", current_error_message},
                       {"currentSeverity", current_error_severity},
                       {"currentTsMs",
                        current_error_ts_ms.has_value() ? json(current_error_ts_ms.value()) : json(nullptr)}}},
    };
    {
      f8::cppsdk::generated::F8MonitorSnapshot parsed;
      f8::cppsdk::generated::ParseError perr;
      if (!f8::cppsdk::generated::parse_F8MonitorSnapshot(snapshot, parsed, perr)) {
        monitor_record_error("MONITOR_SCHEMA_INVALID", perr.message.empty() ? "invalid monitor snapshot" : perr.message, ts);
        continue;
      }
    }
    (void)runtime_publish_data(cfg_.service_id, "monitor", snapshot, ts);
  }
}

bool ServiceBus::is_active() const {
  return active_.load(std::memory_order_acquire);
}

void ServiceBus::on_activate(const json& meta) {
  set_active_local(true, meta, "cmd");
}
void ServiceBus::on_deactivate(const json& meta) {
  set_active_local(false, meta, "cmd");
}
void ServiceBus::on_set_active(bool active, const json& meta) {
  set_active_local(active, meta, "cmd");
}

bool ServiceBus::on_set_state(const std::string& node_id, const std::string& field, const json& value, const json& meta,
                              std::string& error_code, std::string& error_message) {
  if (node_id == cfg_.service_id && field == "active") {
    if (!value.is_boolean()) {
      error_code = "INVALID_ARGS";
      error_message = "active must be boolean";
      return false;
    }
    set_active_local(value.get<bool>(), meta, "endpoint");
    return true;
  }

  std::string node_id_s;
  try {
    node_id_s = ensure_token(node_id, "node_id");
  } catch (const std::exception& exc) {
    error_code = "INVALID_ARGS";
    error_message = std::string("invalid nodeId: ") + exc.what();
    return false;
  } catch (...) {
    error_code = "INVALID_ARGS";
    error_message = "invalid nodeId";
    return false;
  }

  std::string field_s = trim_copy(field);
  if (field_s.empty()) {
    error_code = "INVALID_ARGS";
    error_message = "field must be non-empty";
    return false;
  }

  std::string access;
  bool is_hidden_command_input = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = state_access_.find({node_id_s, field_s});
    if (it != state_access_.end()) access = it->second;
    is_hidden_command_input = command_input_bindings_.find({node_id_s, field_s}) != command_input_bindings_.end();
    if (has_rungraph_ && it == state_access_.end()) {
      error_code = "UNKNOWN_FIELD";
      error_message = "unknown state field";
      return false;
    }
  }
  if (!access.empty() && !state_origin_allows_access("external", access)) {
    error_code = "FORBIDDEN";
    error_message = "state field not writable";
    return false;
  }
  if (is_hidden_command_input) {
    if (!publish_state_local(node_id_s, field_s, value, now_ms(), "endpoint", meta, "external", true, true)) {
      error_code = "INTERNAL";
      error_message = "state persistence failed";
      return false;
    }
    error_code.clear();
    error_message.clear();
    return true;
  }

  std::vector<SetStateHandlerNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = set_state_nodes_;
  }
  if (nodes.empty()) {
    if (!publish_state_local(node_id_s, field_s, value, now_ms(), "endpoint", meta, "external", true, true)) {
      error_code = "INTERNAL";
      error_message = "state persistence failed";
      return false;
    }
    return true;
  }

  for (auto* n : nodes) {
    if (!n) continue;
    error_code.clear();
    error_message.clear();
    try {
      if (n->on_set_state(node_id, field, value, meta, error_code, error_message)) {
        return true;
      }
      if (!error_code.empty() || !error_message.empty()) {
        return false;
      }
    } catch (const std::exception& exc) {
      spdlog::error("on_set_state callback failed serviceId={} nodeId={} field={}: {}", cfg_.service_id, node_id,
                    field, exc.what());
      error_code = "INTERNAL_ERROR";
      error_message = exc.what();
      return false;
    } catch (...) {
      spdlog::error("on_set_state callback failed serviceId={} nodeId={} field={}: unknown error", cfg_.service_id,
                    node_id, field);
      error_code = "INTERNAL_ERROR";
      error_message = "on_set_state threw unknown error";
      return false;
    }
  }
  error_code = "NOT_SUPPORTED";
  error_message = "set_state not supported";
  return false;
}

bool ServiceBus::on_set_rungraph(const json& graph_obj, const json& meta, std::string& error_code,
                                 std::string& error_message) {
  error_code.clear();
  error_message.clear();
  try {
    json persisted = graph_obj;
    if (!persisted.contains("meta") || !persisted["meta"].is_object()) {
      persisted["meta"] = json::object();
    }
    if (meta.is_object() && meta.contains("source") && meta["source"].is_string() &&
        !persisted["meta"].contains("source")) {
      persisted["meta"]["source"] = meta["source"];
    }
    persisted["meta"]["ts"] = now_ms();
    apply_rungraph_local(persisted, error_code, error_message);
    if (!error_code.empty()) {
      return false;
    }
    std::string target_fingerprint;
    if (meta.is_object() && meta.contains("targetFingerprint") && meta["targetFingerprint"].is_string()) {
      target_fingerprint = trim_copy(meta["targetFingerprint"].get<std::string>());
    }
    const std::string applied_fingerprint =
        target_fingerprint.empty() ? build_rungraph_deploy_fingerprint(persisted) : target_fingerprint;
    set_rungraph_metadata(persisted.value("graphId", ""), persisted.value("revision", ""), applied_fingerprint);
    const auto bytes = encode_json(persisted);
    (void)runtime_retained_put(rungraph_key(cfg_.service_id), bytes);
  } catch (const std::exception& ex) {
    error_code = "INTERNAL";
    error_message = ex.what();
    return false;
  } catch (...) {
    error_code = "INTERNAL";
    error_message = "unknown error";
    return false;
  }
  std::vector<RungraphHandlerNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = rungraph_nodes_;
  }
  // Best-effort hook calls (mirrors pysdk's rungraph hooks boundary).
  for (auto* n : nodes) {
    if (!n) continue;
    try {
      std::string _code;
      std::string _msg;
      (void)n->on_set_rungraph(graph_obj, meta, _code, _msg);
    } catch (const std::exception& exc) {
      spdlog::warn("rungraph hook failed serviceId={}: {}", cfg_.service_id, exc.what());
    } catch (...) {
      spdlog::warn("rungraph hook failed serviceId={}: unknown error", cfg_.service_id);
    }
  }
  return true;
}

ServiceBus::_RungraphMetadata ServiceBus::rungraph_metadata_snapshot() const {
  std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
  return _RungraphMetadata{rungraph_graph_id_, rungraph_revision_, rungraph_fingerprint_};
}

void ServiceBus::set_rungraph_metadata(std::string graph_id, std::string revision, std::string fingerprint) {
  std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
  rungraph_graph_id_ = std::move(graph_id);
  rungraph_revision_ = std::move(revision);
  rungraph_fingerprint_ = std::move(fingerprint);
}

bool ServiceBus::submit_rungraph(const json& graph_obj, const json& meta, const std::string& req_id,
                                 std::string& error_code, std::string& error_message) {
  error_code.clear();
  error_message.clear();

  std::string req_id_s = trim_copy(req_id);
  if (req_id_s.empty()) {
    req_id_s = new_control_req_id();
  }

  std::string source = "control";
  std::string target_fingerprint;
  if (meta.is_object() && meta.contains("source") && meta["source"].is_string()) {
    source = trim_copy(meta["source"].get<std::string>());
    if (source.empty()) {
      source = "control";
    }
  }
  if (meta.is_object() && meta.contains("targetFingerprint") && meta["targetFingerprint"].is_string()) {
    target_fingerprint = trim_copy(meta["targetFingerprint"].get<std::string>());
  }
  const bool force_apply =
      meta.is_object() && meta.contains("forceApply") && meta["forceApply"].is_boolean() && meta["forceApply"].get<bool>();
  if (target_fingerprint.empty()) {
    target_fingerprint = build_rungraph_deploy_fingerprint(graph_obj);
  }
  bool publish_applied = false;
  bool publish_accepted = false;
  bool notify_worker = false;

  try {
    {
      std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
      const auto existing_req = rungraph_req_fingerprints_.find(req_id_s);
      if (existing_req != rungraph_req_fingerprints_.end() && existing_req->second != target_fingerprint) {
        error_code = "INVALID_ARGS";
        error_message = "reqId already used for a different rungraph fingerprint";
        return false;
      }
      rungraph_req_fingerprints_[req_id_s] = target_fingerprint;
      const std::string current_fingerprint = rungraph_fingerprint_;
      if (!force_apply && !current_fingerprint.empty() && current_fingerprint == target_fingerprint) {
        publish_applied = true;
      } else if (auto aliases_it = rungraph_inflight_aliases_.find(target_fingerprint);
                 aliases_it != rungraph_inflight_aliases_.end()) {
        aliases_it->second.insert(req_id_s);
        publish_accepted = true;
      } else if (!rungraph_apply_running_) {
        error_code = "NOT_READY";
        error_message = "rungraph apply worker is not running";
        return false;
      } else {
        rungraph_inflight_aliases_[target_fingerprint].insert(req_id_s);
        notify_worker = true;
        rungraph_apply_queue_.push_back(_RungraphApplyRequest{graph_obj, meta, req_id_s, source, target_fingerprint});
      }
    }
    if (notify_worker) {
      rungraph_apply_cv_.notify_one();
    }
  } catch (const std::exception& exc) {
    error_code = "INTERNAL";
    error_message = exc.what();
    return false;
  } catch (...) {
    error_code = "INTERNAL";
    error_message = "failed to start rungraph apply task";
    return false;
  }
  if (publish_applied) {
    publish_rungraph_deploy_status(graph_obj, req_id_s, "applied", source, target_fingerprint, target_fingerprint);
  }
  if (publish_accepted) {
    publish_rungraph_deploy_status(graph_obj, req_id_s, "accepted", source, target_fingerprint);
  }
  return true;
}

void ServiceBus::start_rungraph_apply_worker() {
  stop_rungraph_apply_worker();
  {
    std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
    rungraph_apply_queue_.clear();
    rungraph_apply_stop_requested_ = false;
    rungraph_apply_running_ = true;
  }
  rungraph_apply_thread_ = std::thread([this]() { rungraph_apply_worker_loop(); });
}

void ServiceBus::stop_rungraph_apply_worker() {
  {
    std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
    rungraph_apply_stop_requested_ = true;
    rungraph_apply_running_ = false;
    rungraph_apply_queue_.clear();
    rungraph_inflight_aliases_.clear();
  }
  rungraph_apply_cv_.notify_all();
  if (rungraph_apply_thread_.joinable()) {
    try {
      rungraph_apply_thread_.join();
    } catch (const std::system_error& exc) {
      spdlog::warn("rungraph apply worker join failed serviceId={}: {}", cfg_.service_id, exc.what());
    }
  }
}

void ServiceBus::rungraph_apply_worker_loop() {
  while (true) {
    _RungraphApplyRequest request;
    {
      std::unique_lock<std::mutex> lock(rungraph_apply_mu_);
      rungraph_apply_cv_.wait(
          lock, [this]() { return rungraph_apply_stop_requested_ || !rungraph_apply_queue_.empty(); });
      if (rungraph_apply_stop_requested_) {
        return;
      }
      request = std::move(rungraph_apply_queue_.front());
      rungraph_apply_queue_.pop_front();
    }
    run_rungraph_apply_worker(std::move(request.graph_obj), std::move(request.meta),
                              std::move(request.target_fingerprint), std::move(request.source));
  }
}

void ServiceBus::run_rungraph_apply_worker(json graph_obj, json meta, std::string target_fingerprint,
                                           std::string source) {
  std::vector<std::string> aliases;
  {
    std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
    const auto it = rungraph_inflight_aliases_.find(target_fingerprint);
    if (it != rungraph_inflight_aliases_.end()) {
      aliases.assign(it->second.begin(), it->second.end());
    }
  }
  publish_rungraph_deploy_status_for_aliases(graph_obj, aliases, "accepted", source, target_fingerprint);
  publish_rungraph_deploy_status_for_aliases(graph_obj, aliases, "applying", source, target_fingerprint);

  auto collect_and_clear_aliases = [this, &target_fingerprint]() {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lock(rungraph_apply_mu_);
    const auto it = rungraph_inflight_aliases_.find(target_fingerprint);
    if (it != rungraph_inflight_aliases_.end()) {
      out.assign(it->second.begin(), it->second.end());
      rungraph_inflight_aliases_.erase(it);
    }
    return out;
  };

  std::string error_code;
  std::string error_message;
  const auto apply_started = std::chrono::steady_clock::now();
  spdlog::info("rungraph async apply begin serviceId={} fingerprint={} aliases={}", cfg_.service_id,
               target_fingerprint.substr(0, 16), aliases.size());

  bool ok = false;
  try {
    ok = on_set_rungraph(graph_obj, meta, error_code, error_message);
  } catch (const std::exception& exc) {
    ok = false;
    error_code = "INTERNAL_ERROR";
    error_message = std::string("on_set_rungraph threw: ") + exc.what();
    spdlog::error("rungraph async apply threw serviceId={} fingerprint={}: {}", cfg_.service_id,
                  target_fingerprint.substr(0, 16), exc.what());
  } catch (...) {
    ok = false;
    error_code = "INTERNAL_ERROR";
    error_message = "on_set_rungraph threw unknown error";
    spdlog::error("rungraph async apply threw serviceId={} fingerprint={}: unknown error", cfg_.service_id,
                  target_fingerprint.substr(0, 16));
  }
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - apply_started).count();
  if (!ok) {
    const std::string message = error_message.empty() ? error_code : error_message;
    aliases = collect_and_clear_aliases();
    publish_rungraph_deploy_status_for_aliases(graph_obj, aliases, "failed", source, target_fingerprint, "", message);
    monitor_record_error("RUNGRAPH_APPLY_FAILED", message);
    spdlog::error("rungraph async apply failed serviceId={} fingerprint={} elapsedMs={:.1f} code={} message={}",
                  cfg_.service_id, target_fingerprint.substr(0, 16), elapsed_ms, error_code, error_message);
    return;
  }
  const _RungraphMetadata rungraph_metadata = rungraph_metadata_snapshot();
  const std::string applied_fingerprint =
      rungraph_metadata.fingerprint.empty() ? target_fingerprint : rungraph_metadata.fingerprint;
  aliases = collect_and_clear_aliases();
  publish_rungraph_deploy_status_for_aliases(graph_obj, aliases, "applied", source, target_fingerprint,
                                             applied_fingerprint);
  spdlog::info("rungraph async apply applied serviceId={} fingerprint={} elapsedMs={:.1f}", cfg_.service_id,
               target_fingerprint.substr(0, 16), elapsed_ms);
}

void ServiceBus::publish_rungraph_deploy_status(const json& graph_obj, const std::string& req_id,
                                                const std::string& phase, const std::string& source,
                                                const std::string& target_fingerprint,
                                                const std::string& applied_fingerprint,
                                                const std::string& error_message) {
  try {
    const std::string graph_id = graph_obj.is_object() ? graph_obj.value("graphId", "") : "";
    const std::string revision = graph_obj.is_object() ? graph_obj.value("revision", "") : "";
    const std::string phase_s = trim_copy(phase);
    json payload = json::object();
    payload["schemaVersion"] = "f8.rungraphDeployStatus/2";
    payload["serviceId"] = cfg_.service_id;
    payload["reqId"] = req_id;
    payload["graphId"] = graph_id;
    payload["revision"] = revision;
    payload["phase"] = phase_s;
    payload["ok"] = phase_s == "applied";
    payload["source"] = source;
    payload["errorMessage"] = error_message;
    payload["ts"] = now_ms();
    payload["targetFingerprint"] = target_fingerprint;
    payload["appliedFingerprint"] = applied_fingerprint;
    payload["runtimeInstanceId"] = runtime_instance_id_;

    const auto raw = encode_json(payload);
    const std::array<std::string, 2> keys = {
        rungraph_deploy_status_key(cfg_.service_id),
        rungraph_deploy_request_status_key(cfg_.service_id, req_id),
    };
    for (const auto& key : keys) {
      try {
        if (!runtime_retained_put(key, raw)) {
          spdlog::warn("publish rungraph deploy status returned false serviceId={} reqId={} key={}", cfg_.service_id,
                       req_id, key);
        }
      } catch (const std::exception& exc) {
        spdlog::warn("publish rungraph deploy status failed serviceId={} reqId={} key={}: {}", cfg_.service_id,
                     req_id, key, exc.what());
      } catch (...) {
        spdlog::warn("publish rungraph deploy status failed serviceId={} reqId={} key={}: unknown error",
                     cfg_.service_id, req_id, key);
      }
    }
  } catch (const std::exception& exc) {
    spdlog::warn("build rungraph deploy status failed serviceId={} reqId={}: {}", cfg_.service_id, req_id, exc.what());
  } catch (...) {
    spdlog::warn("build rungraph deploy status failed serviceId={} reqId={}: unknown error", cfg_.service_id, req_id);
  }
}

void ServiceBus::publish_rungraph_deploy_status_for_aliases(const json& graph_obj,
                                                            const std::vector<std::string>& req_ids,
                                                            const std::string& phase, const std::string& source,
                                                            const std::string& target_fingerprint,
                                                            const std::string& applied_fingerprint,
                                                            const std::string& error_message) {
  for (const auto& req_id : req_ids) {
    publish_rungraph_deploy_status(graph_obj, req_id, phase, source, target_fingerprint, applied_fingerprint,
                                   error_message);
  }
}

void ServiceBus::rebuild_command_bindings_locked() {
  command_input_bindings_.clear();
  command_output_bindings_.clear();
  command_hidden_fields_.clear();
  for (const auto& entry : command_specs_by_node_) {
    const auto parsed = parse_command_bindings_from_spec(entry.second, cfg_.service_id);
    for (const auto& binding_in : parsed) {
      _CommandBinding binding;
      binding.node_id = binding_in.node_id;
      binding.call = binding_in.call;
      binding.input_field = binding_in.input_field;
      binding.output_field = binding_in.output_field;
      binding.param_names = binding_in.param_names;
      command_input_bindings_[{binding.node_id, binding.input_field}] = binding;
      command_output_bindings_[binding.call] = binding;
      command_hidden_fields_.insert({binding.node_id, binding.input_field});
      command_hidden_fields_.insert({binding.node_id, binding.output_field});
    }
  }
}

bool ServiceBus::dispatch_command_call(const std::string& call, const json& args, const json& meta, json& result,
                                       std::string& error_code, std::string& error_message) {
  std::vector<CommandableNode*> nodes;
  {
    std::lock_guard<std::mutex> lock(handlers_mu_);
    nodes = command_nodes_;
  }
  for (auto* n : nodes) {
    if (!n) continue;
    error_code.clear();
    error_message.clear();
    try {
      if (n->on_command(call, args, meta, result, error_code, error_message)) {
        return true;
      }
      if (!error_code.empty() || !error_message.empty()) {
        return false;
      }
    } catch (const std::exception& exc) {
      spdlog::error("on_command callback failed serviceId={} call={}: {}", cfg_.service_id, call, exc.what());
      error_code = "INTERNAL_ERROR";
      error_message = exc.what();
      monitor_record_error("INTERNAL_ERROR", std::string("on_command threw: ") + exc.what());
      return false;
    } catch (...) {
      spdlog::error("on_command callback failed serviceId={} call={}: unknown error", cfg_.service_id, call);
      error_code = "INTERNAL_ERROR";
      error_message = "on_command threw unknown error";
      monitor_record_error("INTERNAL_ERROR", "on_command threw unknown error");
      return false;
    }
  }
  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void ServiceBus::write_command_output(const std::string& node_id, const std::string& call, const json& result,
                                      std::int64_t ts_ms, const json& meta) {
  _CommandBinding binding;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = command_output_bindings_.find(call);
    if (it == command_output_bindings_.end()) return;
    binding = it->second;
  }
  json out_meta = meta.is_object() ? meta : json::object();
  out_meta["command"] = call;
  publish_state_local(node_id.empty() ? binding.node_id : node_id, binding.output_field, result,
                      ts_ms > 0 ? ts_ms : now_ms(), "cmd", out_meta, "runtime", true, true);
}

void ServiceBus::schedule_command_input_dispatch(const std::string& node_id, const std::string& field, const json& value,
                                                 std::int64_t ts_ms, const json& meta) {
  bool should_post = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto& dispatch = command_dispatch_[{node_id, field}];
    dispatch.latest_value = value;
    dispatch.latest_ts_ms = ts_ms;
    dispatch.latest_meta = meta.is_object() ? meta : json::object();
    dispatch.version += 1;
    if (!dispatch.running) {
      dispatch.running = true;
      should_post = true;
    }
  }
  if (!should_post) return;
  main_thread_.post([this, node_id, field]() { run_command_input_dispatch(node_id, field); });
}

void ServiceBus::run_command_input_dispatch(const std::string& node_id, const std::string& field) {
  while (true) {
    _CommandBinding binding;
    _CommandDispatchState dispatch;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      const auto it_binding = command_input_bindings_.find({node_id, field});
      const auto it_dispatch = command_dispatch_.find({node_id, field});
      if (it_binding == command_input_bindings_.end() || it_dispatch == command_dispatch_.end()) {
        if (it_dispatch != command_dispatch_.end()) it_dispatch->second.running = false;
        return;
      }
      binding = it_binding->second;
      dispatch = it_dispatch->second;
    }

    const json args = map_command_args(dispatch.latest_value, binding.param_names);
    json call_meta = dispatch.latest_meta.is_object() ? dispatch.latest_meta : json::object();
    call_meta["commandInputField"] = field;
    call_meta["source"] = "cmd";

    json result = json::object();
    std::string error_code;
    std::string error_message;
    const bool ok = dispatch_command_call(binding.call, args, call_meta, result, error_code, error_message);
    if (ok) {
      write_command_output(binding.node_id, binding.call, result, dispatch.latest_ts_ms, call_meta);
    } else if (!error_code.empty() || !error_message.empty()) {
      monitor_record_error(error_code.empty() ? "COMMAND_FAILED" : error_code,
                           error_message.empty() ? ("command failed: " + binding.call) : error_message,
                           dispatch.latest_ts_ms);
    }

    bool has_newer = false;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      auto it_dispatch = command_dispatch_.find({node_id, field});
      if (it_dispatch == command_dispatch_.end()) {
        return;
      }
      has_newer = it_dispatch->second.version != dispatch.version;
      if (!has_newer) {
        it_dispatch->second.running = false;
        return;
      }
    }
  }
}

bool ServiceBus::on_command(const std::string& call, const json& args, const json& meta, json& result,
                            std::string& error_code, std::string& error_message) {
  if (call == "terminate" || call == "quit") {
    spdlog::info("service_bus terminate requested serviceId={}", cfg_.service_id);
    terminate_.store(true, std::memory_order_release);
    term_cv_.notify_all();
    result = json::object();
    result["terminating"] = true;
    return true;
  }
  const bool ok = dispatch_command_call(call, args, meta, result, error_code, error_message);
  if (ok) {
    write_command_output(cfg_.service_id, call, result, now_ms(), meta);
  }
  return ok;
}

void ServiceBus::load_active_from_retained() {
  try {
    const auto key = zenoh_state_key(cfg_.service_id, cfg_.service_id, "active");
    const auto raw = runtime_retained_get(key);
    if (!raw.has_value()) {
      return;
    }
    json payload = json::object();
    if (!decode_json(raw->data(), raw->size(), payload)) {
      return;
    }
    if (!payload.is_object() || !payload.contains("value")) {
      return;
    }
    const json v = payload["value"];
    if (!v.is_boolean()) {
      return;
    }
    set_active_local(v.get<bool>(), json::object({{"init", true}}), "kv");
  } catch (const std::exception& exc) {
    spdlog::warn("load active state failed serviceId={}: {}", cfg_.service_id, exc.what());
  } catch (...) {
    spdlog::warn("load active state failed serviceId={}: unknown error", cfg_.service_id);
  }
}

bool ServiceBus::emit_data(const std::string& from_node_id, const std::string& port_id, const json& value,
                           std::int64_t ts_ms) {
  if (!active()) return false;
  const std::string node = ensure_token(from_node_id, "from_node_id");
  const std::string port = ensure_token(port_id, "port_id");
  const std::int64_t now_ts = now_ms();
  monitor_record_processed(port, ts_ms, now_ts);
  return runtime_publish_data(node, port, value, ts_ms);
}

std::optional<json> ServiceBus::pull_data(const std::string& node_id, const std::string& port_id) {
  return pull_data(node_id, port_id, 0);
}

std::optional<json> ServiceBus::pull_data(const std::string& node_id, const std::string& port_id,
                                          std::int64_t ctx_id) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string pid = ensure_token(port_id, "port_id");

  std::shared_ptr<_InputBuffer> buf_ptr;
  DataPullResolver resolver;
  {
    std::lock_guard<std::mutex> lock(data_mu_);
    const auto it = data_inputs_.find({nid, pid});
    if (it != data_inputs_.end()) {
      buf_ptr = it->second;
    }
    resolver = data_pull_resolver_;
  }

  if (buf_ptr) {
    const std::int64_t now = now_ms();
    auto& mut = *buf_ptr;
    std::lock_guard<std::mutex> lock(mut.mu);
    const bool timed_out = mut.timeout_ms > 0 && mut.last_seen_ts_ms > 0 && (now - mut.last_seen_ts_ms) > mut.timeout_ms;
    if (!timed_out && mut.strategy == EdgeStrategy::kQueue) {
      if (!mut.queue.empty()) {
        const auto& sample = mut.queue.front();
        json v = sample.first ? *sample.first : json(nullptr);
        if (sample.second > 0) {
          monitor_record_wait_ms(static_cast<double>(std::max<std::int64_t>(0, now - sample.second)));
        }
        mut.queue.pop_front();
        return v;
      }
    }

    // latest
    if (!timed_out && !mut.queue.empty()) {
      const auto& sample = mut.queue.back();
      json v = sample.first ? *sample.first : json(nullptr);
      if (sample.second > 0) {
        monitor_record_wait_ms(static_cast<double>(std::max<std::int64_t>(0, now - sample.second)));
      }
      mut.queue.clear();
      return v;
    }
    if (!timed_out && mut.last_seen_ts_ms > 0 && mut.last_seen_value) {
      return *mut.last_seen_value;
    }
  }

  if (resolver) {
    return resolver(nid, pid, ctx_id);
  }
  return std::nullopt;
}

void ServiceBus::set_data_pull_resolver(DataPullResolver resolver) {
  std::lock_guard<std::mutex> lock(data_mu_);
  data_pull_resolver_ = std::move(resolver);
}

void ServiceBus::push_data_input_for_local_test(const std::string& node_id, const std::string& port_id,
                                                const json& value, std::int64_t ts_ms) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string pid = ensure_token(port_id, "port_id");
  const auto value_ptr = std::make_shared<const json>(value);
  std::shared_ptr<_InputBuffer> buf_ptr;
  {
    std::lock_guard<std::mutex> lock(data_mu_);
    auto it = data_inputs_.find({nid, pid});
    if (it == data_inputs_.end()) {
      it = data_inputs_.emplace(_NodePortKey{nid, pid}, std::make_shared<_InputBuffer>()).first;
    }
    buf_ptr = it->second;
  }
  if (!buf_ptr) return;
  std::lock_guard<std::mutex> lock(buf_ptr->mu);
  buf_ptr->last_seen_value = value_ptr;
  buf_ptr->last_seen_ts_ms = ts_ms > 0 ? ts_ms : now_ms();
  if (buf_ptr->strategy == EdgeStrategy::kLatest) {
    buf_ptr->queue.clear();
  }
  buf_ptr->queue.emplace_back(value_ptr, buf_ptr->last_seen_ts_ms);
}

std::optional<std::string> ServiceBus::data_input_zenoh_key(const std::string& node_id,
                                                            const std::string& port_id) const {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string pid = ensure_token(port_id, "port_id");
  std::lock_guard<std::mutex> lock(data_mu_);
  const auto it = data_input_stream_keys_.find({nid, pid});
  if (it == data_input_stream_keys_.end() || it->second.empty()) {
    return std::nullopt;
  }
  return it->second;
}

ServiceBus::StateRead ServiceBus::get_state(const std::string& node_id, const std::string& field) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string f = field;
  if (f.empty()) {
    return StateRead{false, json(nullptr), std::nullopt};
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    auto it = state_cache_.find({nid, f});
    if (it != state_cache_.end()) {
      return StateRead{true, it->second.first, it->second.second};
    }
  }

  const auto key = zenoh_state_key(cfg_.service_id, nid, f);
  auto raw = runtime_retained_get(key);
  if (!raw.has_value()) {
    return StateRead{false, json(nullptr), std::nullopt};
  }
  json payload = json::object();
  (void)decode_json(raw->data(), raw->size(), payload);
  if (!payload.is_object() || !payload.contains("value")) {
    return StateRead{true, json::binary(*raw), std::int64_t{0}};
  }
  const json v = payload["value"];
  const std::int64_t ts = coerce_inbound_ts_ms(payload, 0);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_cache_[{nid, f}] = {v, ts};
  }
  return StateRead{true, v, ts};
}

bool ServiceBus::publish_state(const std::string& node_id, const std::string& field, const json& value,
                               const std::string& source, const json& meta, std::int64_t ts_ms,
                               const std::string& origin) {
  try {
    return publish_state_local(node_id, field, value, ts_ms > 0 ? ts_ms : now_ms(), source, meta, origin, false, false);
  } catch (const std::exception& exc) {
    spdlog::warn("publish_state failed serviceId={} nodeId={} field={}: {}", cfg_.service_id, node_id, field,
                 exc.what());
    return false;
  } catch (...) {
    spdlog::warn("publish_state failed serviceId={} nodeId={} field={}: unknown error", cfg_.service_id, node_id,
                 field);
    return false;
  }
}

bool ServiceBus::publish_state_from_external(const std::string& node_id, const std::string& field, const json& value,
                                             const json& meta, std::int64_t ts_ms) {
  try {
    return publish_state_local(node_id, field, value, ts_ms > 0 ? ts_ms : now_ms(), "endpoint", meta, "external", true,
                               true);
  } catch (const std::exception& exc) {
    spdlog::warn("publish_state_from_external failed serviceId={} nodeId={} field={}: {}", cfg_.service_id, node_id,
                 field, exc.what());
    return false;
  } catch (...) {
    spdlog::warn("publish_state_from_external failed serviceId={} nodeId={} field={}: unknown error", cfg_.service_id,
                 node_id, field);
    return false;
  }
}

std::int64_t ServiceBus::rungraph_ts_ms(const f8::cppsdk::generated::F8RuntimeGraph& graph) const {
  if (!graph.meta.has_value()) {
    return 0;
  }
  const std::optional<std::int64_t>& ts = graph.meta->ts;
  if (!ts.has_value()) {
    return 0;
  }
  return ts.value();
}

bool ServiceBus::should_apply_rungraph_state_value(const std::string& node_id, const std::string& field,
                                                   const json& value, std::int64_t rungraph_ts) {
  if (rungraph_ts <= 0) {
    return true;
  }

  const auto existing = get_state(node_id, field);
  if (!existing.found) {
    return true;
  }

  try {
    if (existing.value == value) {
      return false;
    }
  } catch (const std::exception& exc) {
    spdlog::warn("rungraph state reconcile compare failed serviceId={} nodeId={} field={}: {}", cfg_.service_id,
                 node_id, field, exc.what());
  }

  if (existing.ts_ms.has_value() && existing.ts_ms.value() >= rungraph_ts) {
    return false;
  }
  return true;
}

bool ServiceBus::build_rungraph_state_routing_plan(const f8::cppsdk::generated::F8RuntimeGraph& graph,
                                                   _RungraphStateRoutingPlan& plan,
                                                   std::string& error_code,
                                                   std::string& error_message) const {
  using namespace f8::cppsdk::generated;

  plan = _RungraphStateRoutingPlan{};
  const std::string sid = cfg_.service_id;

  // Build access map and validate rungraph-provided stateValues.
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (n.serviceId != sid) continue;
    if (n.nodeId.empty()) {
      error_code = "INVALID_RUNGRAPH";
      error_message = "missing nodeId";
      return false;
    }

    std::unordered_map<std::string, std::string> access_by_name;
    if (n.stateFields.has_value()) {
      for (const auto& sf : n.stateFields.value()) {
        const std::string name = sf.name;
        if (name.empty()) continue;
        const std::string access_s = access_to_string(sf.access);
        plan.state_access[{n.nodeId, name}] = access_s;
        access_by_name[name] = access_s;
      }
    }

    if (n.stateValues.is_object()) {
      for (auto it = n.stateValues.begin(); it != n.stateValues.end(); ++it) {
        const std::string field = it.key();
        const auto access_it = access_by_name.find(field);
        if (access_it == access_by_name.end()) {
          error_code = "INVALID_RUNGRAPH";
          error_message = "unknown state value: " + n.nodeId + "." + field;
          return false;
        }
        if (access_it->second == "ro") {
          error_code = "INVALID_RUNGRAPH";
          error_message = "read-only state cannot be set by rungraph: " + n.nodeId + "." + field;
          return false;
        }
      }
    }
  }

  std::unordered_set<_RemoteStateKey, _RemoteStateKeyHash> cross_state_initial_read_set;

  for (const auto& e : graph.edges.value_or(std::vector<F8Edge>{})) {
    if (e.kind != F8EdgeKindEnum::state) continue;
    const std::string from_sid = e.fromServiceId;
    const std::string to_sid = e.toServiceId;
    if (to_sid != sid) continue;

    std::string from_node = e.fromOperatorId.value_or("");
    if (from_node.empty()) from_node = from_sid;
    std::string to_node = e.toOperatorId.value_or("");
    if (to_node.empty()) to_node = sid;
    const std::string from_field = e.fromPort;
    const std::string to_field = e.toPort;
    if (from_sid.empty() || to_node.empty() || from_node.empty() || from_field.empty() || to_field.empty()) continue;

    // Pre-filter to only writable targets for external propagation to reduce per-update overhead.
    const auto access_it = plan.state_access.find({to_node, to_field});
    if (access_it == plan.state_access.end()) continue;
    if (access_it->second == "ro") continue;

    if (from_sid == sid) {
      plan.intra_state_out[{from_node, from_field}].push_back({to_node, to_field});
      continue;
    }

    const _RemoteStateKey remote_state_key{from_sid, from_node, from_field};
    plan.cross_state_in[remote_state_key].push_back({to_node, to_field});
    plan.cross_state_targets.insert({to_node, to_field});
    if (cross_state_initial_read_set.insert(remote_state_key).second) {
      plan.cross_state_initial_reads.push_back(remote_state_key);
    }
  }

  return true;
}

std::unordered_map<ServiceBus::_NodeFieldKey, std::string, ServiceBus::_NodeFieldKeyHash>
ServiceBus::install_rungraph_state_routing_plan(_RungraphStateRoutingPlan plan) {
  std::unordered_map<_NodeFieldKey, std::string, _NodeFieldKeyHash> state_access_snapshot;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_access_ = std::move(plan.state_access);
    intra_state_out_ = std::move(plan.intra_state_out);
    cross_state_in_ = std::move(plan.cross_state_in);
    cross_state_targets_ = std::move(plan.cross_state_targets);
    rebuild_command_bindings_locked();
    has_rungraph_ = true;
    state_access_snapshot = state_access_;
  }
  return state_access_snapshot;
}

void ServiceBus::sync_rungraph_peer_state_watches(const std::vector<_RemoteStateKey>& cross_state_initial_reads) {
  std::unordered_set<std::string> want_peers;
  for (const auto& remote_state_key : cross_state_initial_reads) {
    want_peers.insert(remote_state_key.peer_service_id);
  }

  std::vector<std::unique_ptr<RuntimeSubscription>> removed_peer_subscriptions;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    for (auto it = peer_state_subs_by_service_id_.begin(); it != peer_state_subs_by_service_id_.end();) {
      if (want_peers.find(it->first) != want_peers.end()) {
        ++it;
        continue;
      }
      removed_peer_subscriptions.push_back(std::move(it->second));
      it = peer_state_subs_by_service_id_.erase(it);
    }
  }
  stop_runtime_subscriptions(removed_peer_subscriptions, cfg_.service_id, "peer state subscription stop");

  for (const auto& peer : want_peers) {
    bool has_peer = false;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      has_peer = (peer_state_subs_by_service_id_.find(peer) != peer_state_subs_by_service_id_.end());
    }
    if (has_peer) continue;
    if (!runtime_transport_) {
      spdlog::warn("peer state subscription skipped without runtime transport serviceId={} peer={}", cfg_.service_id,
                   peer);
      continue;
    }

    const std::string key_expr = zenoh_state_path_pattern(peer, "nodes.>");
    auto sub = runtime_transport_->retained_watch(
        key_expr,
        [this, peer](const std::string& key, const RuntimeBytes& bytes) {
          handle_peer_state_payload(peer, key, bytes);
        });
    if (!sub || !sub->valid()) {
      spdlog::warn("peer state subscription failed serviceId={} peer={} keyExpr={}", cfg_.service_id, peer, key_expr);
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      peer_state_subs_by_service_id_[peer] = std::move(sub);
    }
    if (state_debug_enabled()) {
      spdlog::info("state_debug[{}] cross_state_watch_started peer={} keyExpr={}", cfg_.service_id, peer, key_expr);
    }
  }
}

void ServiceBus::apply_rungraph_state_values(
    const f8::cppsdk::generated::F8RuntimeGraph& graph,
    const std::unordered_set<_NodeFieldKey, _NodeFieldKeyHash>& cross_state_targets,
    std::int64_t rungraph_ts) {
  using namespace f8::cppsdk::generated;

  const std::string sid = cfg_.service_id;
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (n.serviceId != sid) continue;
    const std::string node_id = n.nodeId;
    if (!n.stateValues.is_object()) continue;

    for (auto it = n.stateValues.begin(); it != n.stateValues.end(); ++it) {
      const std::string field = it.key();
      const json value = it.value();

      // Cross-service state edges are directional: downstream follows upstream.
      // Do not apply rungraph stateValues to fields that are cross-state targets,
      // otherwise local UI defaults (often empty) can clobber remote-propagated values.
      if (cross_state_targets.find(_NodeFieldKey{node_id, field}) != cross_state_targets.end()) {
        if (state_debug_enabled()) {
          spdlog::info("state_debug[{}] cross_state_skip_rungraph node={}.{} value={}", cfg_.service_id, node_id, field,
                       value.dump());
        }
        continue;
      }

      if (node_id == sid && field == "active") {
        if (value.is_boolean()) {
          if (!should_apply_rungraph_state_value(node_id, field, value, rungraph_ts)) continue;
          set_active_local(value.get<bool>(), json{{"via", "rungraph"}, {"rungraphReconcile", true}}, "rungraph");
        } else {
          spdlog::warn("rungraph active reconcile skipped non-boolean value serviceId={}", cfg_.service_id);
        }
        continue;
      }

      if (!should_apply_rungraph_state_value(node_id, field, value, rungraph_ts)) continue;
      publish_state_local(node_id, field, value, rungraph_ts > 0 ? rungraph_ts : now_ms(), "rungraph",
                          json{{"via", "rungraph"}, {"rungraphReconcile", true}}, "rungraph",
                          true, false);
    }
  }
}

void ServiceBus::initial_sync_intra_state_edges(
    const f8::cppsdk::generated::F8RuntimeGraph& graph,
    const std::unordered_map<_NodeFieldKey, std::string, _NodeFieldKeyHash>& state_access_snapshot) {
  using namespace f8::cppsdk::generated;

  const std::string sid = cfg_.service_id;
  std::unordered_map<_NodeFieldKey, std::vector<_NodeFieldKey>, _NodeFieldKeyHash> out_edges;
  std::unordered_set<_NodeFieldKey, _NodeFieldKeyHash> inbound;
  std::unordered_set<_NodeFieldKey, _NodeFieldKeyHash> nodes_set;
  std::vector<_NodeFieldKey> nodes;

  auto add_node = [&](const _NodeFieldKey& key) {
    if (nodes_set.find(key) != nodes_set.end()) return;
    nodes_set.insert(key);
    nodes.push_back(key);
  };

  for (const auto& e : graph.edges.value_or(std::vector<F8Edge>{})) {
    if (e.kind != F8EdgeKindEnum::state) continue;
    const std::string from_sid = e.fromServiceId;
    const std::string to_sid = e.toServiceId;
    if (from_sid != sid || to_sid != sid) continue;

    std::string from_node = e.fromOperatorId.value_or("");
    if (from_node.empty()) from_node = from_sid;
    std::string to_node = e.toOperatorId.value_or("");
    if (to_node.empty()) to_node = sid;
    const std::string from_field = e.fromPort;
    const std::string to_field = e.toPort;
    if (from_node.empty() || to_node.empty() || from_field.empty() || to_field.empty()) continue;

    const auto access_it = state_access_snapshot.find({to_node, to_field});
    if (access_it == state_access_snapshot.end()) continue;
    if (access_it->second != "rw" && access_it->second != "wo") continue;

    _NodeFieldKey from_key{from_node, from_field};
    _NodeFieldKey to_key{to_node, to_field};
    out_edges[from_key].push_back(to_key);
    inbound.insert(to_key);
    add_node(from_key);
    add_node(to_key);
  }

  if (out_edges.empty()) {
    return;
  }

  std::vector<_NodeFieldKey> roots;
  roots.reserve(nodes.size());
  for (const auto& key : nodes) {
    if (inbound.find(key) == inbound.end()) {
      roots.push_back(key);
    }
  }

  const std::int64_t init_ts = now_ms();
  for (const auto& root : roots) {
    const auto root_state = get_state(root.node_id, root.field);
    if (!root_state.found) continue;

    std::vector<std::pair<_NodeFieldKey, json>> queue;
    queue.push_back({root, root_state.value});
    std::size_t head = 0;
    std::unordered_set<_NodeFieldKey, _NodeFieldKeyHash> seen;

    while (head < queue.size()) {
      const auto from_key = queue[head].first;
      const auto from_value = queue[head].second;
      ++head;

      if (seen.find(from_key) != seen.end()) continue;
      seen.insert(from_key);

      const auto edge_it = out_edges.find(from_key);
      if (edge_it == out_edges.end()) continue;

      for (const auto& to_key : edge_it->second) {
        if (seen.find(to_key) != seen.end()) continue;

        const auto to_state = get_state(to_key.node_id, to_key.field);
        if (to_state.found) {
          bool same = false;
          try {
            same = (to_state.value == from_value);
          } catch (const std::exception& exc) {
            spdlog::warn("state edge init compare failed serviceId={} nodeId={} field={}: {}", cfg_.service_id,
                         to_key.node_id, to_key.field, exc.what());
          }
          if (same) {
            queue.push_back({to_key, to_state.value});
            continue;
          }
        }

        publish_state_local(
            to_key.node_id,
            to_key.field,
            from_value,
            init_ts,
            "state_edge_intra_init",
            json{{"fromNodeId", from_key.node_id}, {"fromField", from_key.field}},
            "external",
            true,
            true);

        const auto applied_state = get_state(to_key.node_id, to_key.field);
        if (applied_state.found) {
          queue.push_back({to_key, applied_state.value});
        } else {
          queue.push_back({to_key, from_value});
        }
      }
    }
  }
}

void ServiceBus::seed_rungraph_identity_state(
    const f8::cppsdk::generated::F8RuntimeGraph& graph,
    const std::unordered_map<_NodeFieldKey, std::string, _NodeFieldKeyHash>& state_access_snapshot,
    std::int64_t rungraph_ts) {
  using namespace f8::cppsdk::generated;

  const std::string sid = cfg_.service_id;
  const std::int64_t ts_ms = rungraph_ts > 0 ? rungraph_ts : now_ms();
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (n.serviceId != sid) continue;
    const std::string node_id = n.nodeId;
    const bool has_svc_id = state_access_snapshot.find({node_id, "svcId"}) != state_access_snapshot.end();
    const bool has_operator_id =
        n.operatorClass.has_value() && state_access_snapshot.find({node_id, "operatorId"}) != state_access_snapshot.end();

    if (has_svc_id) {
      publish_state_local(node_id, "svcId", n.serviceId.empty() ? sid : n.serviceId, ts_ms,
                          "system", json{{"builtin", true}}, "system", false, false);
    }
    if (has_operator_id) {
      publish_state_local(node_id, "operatorId", n.nodeId, ts_ms,
                          "system", json{{"builtin", true}}, "system", false, false);
    }
  }
}

void ServiceBus::apply_rungraph_local(const json& graph_obj, std::string& error_code, std::string& error_message) {
  using namespace f8::cppsdk::generated;

  F8RuntimeGraph graph{};
  ParseError perr{};
  if (!parse_F8RuntimeGraph(graph_obj, graph, perr)) {
    error_code = "INVALID_RUNGRAPH";
    error_message = perr.message.empty() ? "invalid rungraph" : perr.message;
    return;
  }
  try {
    validate_state_edges_or_throw(graph);
  } catch (const std::exception& ex) {
    error_code = "INVALID_RUNGRAPH";
    error_message = ex.what();
    return;
  }

  // Service/container nodes require nodeId == serviceId.
  for (const auto& n : graph.nodes.value_or(std::vector<F8RuntimeNode>{})) {
    if (!n.operatorClass.has_value() && n.nodeId != n.serviceId) {
      error_code = "INVALID_RUNGRAPH";
      error_message = "service node requires nodeId == serviceId";
      return;
    }
  }

  _RungraphStateRoutingPlan state_plan;
  if (!build_rungraph_state_routing_plan(graph, state_plan, error_code, error_message)) {
    return;
  }
  const std::vector<_RemoteStateKey> cross_state_initial_reads = state_plan.cross_state_initial_reads;
  const std::unordered_set<_NodeFieldKey, _NodeFieldKeyHash> cross_state_targets_snapshot = state_plan.cross_state_targets;
  const auto state_access_snapshot = install_rungraph_state_routing_plan(std::move(state_plan));

  apply_data_routes_from_rungraph(graph_obj);
  sync_rungraph_peer_state_watches(cross_state_initial_reads);

  // Zenoh retained state history delivers current peer values through the watch itself.
  const std::int64_t rungraph_ts = rungraph_ts_ms(graph);
  apply_rungraph_state_values(graph, cross_state_targets_snapshot, rungraph_ts);
  initial_sync_intra_state_edges(graph, state_access_snapshot);
  seed_rungraph_identity_state(graph, state_access_snapshot, rungraph_ts);
}

bool ServiceBus::publish_state_local(const std::string& node_id, const std::string& field, const json& value,
                                     std::int64_t ts_ms, const std::string& source, const json& meta,
                                     const std::string& origin, bool deliver_local, bool allow_state_fanout) {
  const std::string nid = ensure_token(node_id, "node_id");
  const std::string f = field;
  if (f.empty()) return false;

  // Access enforcement when rungraph is known.
  std::string access;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = state_access_.find({nid, f});
    if (it != state_access_.end()) access = it->second;
    if (has_rungraph_ && it == state_access_.end()) {
      return false;
    }
  }
  if (!access.empty() && !state_origin_allows_access(origin, access)) {
    return false;
  }

  // Value-dedupe after access checks, so a rejected write cannot appear successful.
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = state_cache_.find({nid, f});
    const bool is_command_input = command_input_bindings_.find({nid, f}) != command_input_bindings_.end();
    if (it != state_cache_.end() && !is_command_input && it->second.first == value) {
      return true;
    }
  }

  const json extra = meta.is_object() ? meta : json::object();
  if (!runtime_set_node_state(nid, f, value, source, extra, ts_ms, origin)) {
    spdlog::error("state persistence failed serviceId={} nodeId={} field={}", cfg_.service_id, nid, f);
    return false;
  }
  bool is_command_input = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    state_cache_[{nid, f}] = {value, ts_ms};
    is_command_input = command_input_bindings_.find({nid, f}) != command_input_bindings_.end();
  }
  if (deliver_local) {
    if (is_command_input) {
      schedule_command_input_dispatch(nid, f, value, ts_ms, extra);
      return true;
    }
    main_thread_.post([this, nid, f, value, ts_ms, extra, allow_state_fanout]() {
      deliver_state_local(nid, f, value, ts_ms, extra, allow_state_fanout);
    });
  }
  return true;
}

void ServiceBus::deliver_state_local(const std::string& node_id, const std::string& field, const json& value,
                                     std::int64_t ts_ms, const json& meta, bool allow_state_fanout) {
  std::string hidden_direction;
  const bool is_hidden_command = hidden_command_state_direction(field, hidden_direction);
  if (is_hidden_command && hidden_direction == "in") {
    return;
  }
  std::vector<StatefulNode*> nodes;
  if (!(is_hidden_command && hidden_direction == "out")) {
    {
      std::lock_guard<std::mutex> lock(handlers_mu_);
      nodes = stateful_nodes_;
    }
    for (auto* n : nodes) {
      if (!n) continue;
      try {
        n->on_state(node_id, field, value, ts_ms, meta);
      } catch (const std::exception& exc) {
        spdlog::warn("state callback failed serviceId={} nodeId={} field={}: {}", cfg_.service_id, node_id, field,
                     exc.what());
      } catch (...) {
        spdlog::warn("state callback failed serviceId={} nodeId={} field={}: unknown error", cfg_.service_id, node_id,
                     field);
      }
    }
  }
  if (allow_state_fanout) {
    route_intra_state_edges(node_id, field, value, ts_ms);
  }
}

void ServiceBus::route_intra_state_edges(const std::string& from_node_id, const std::string& from_field,
                                         const json& value, std::int64_t ts_ms) {
  std::vector<_NodeFieldKey> targets;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    const auto it = intra_state_out_.find({from_node_id, from_field});
    if (it != intra_state_out_.end()) {
      targets = it->second;
    }
  }
  if (targets.empty()) return;
  for (const auto& t : targets) {
    publish_state_local(t.node_id, t.field, value, ts_ms, "state_edge_intra", json{{"fromNodeId", from_node_id}, {"fromField", from_field}},
                        "external", true, true);
  }
}

}  // namespace f8::cppsdk
