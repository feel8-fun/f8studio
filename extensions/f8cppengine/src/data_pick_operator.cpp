#include "operator_common.h"

#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "f8cppengine/constants.h"
#include "f8cppsdk/runtime_node.h"
#include "f8cppsdk/runtime_node_registry.h"

namespace f8::cppengine {

using f8::cppsdk::ComputableNode;
using f8::cppsdk::OperatorNode;
using f8::cppsdk::RuntimeNodeRegistry;
using f8::cppsdk::generated::F8RuntimeNode;

namespace {

enum class DataPickPathSegmentKind {
  kKey,
  kIndex,
};

struct DataPickPathSegment {
  DataPickPathSegmentKind kind;
  std::string key;
  std::size_t index = 0;
};

std::string trim_string(const std::string& value) {
  std::size_t first = 0;
  while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first])) != 0) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
    --last;
  }
  return value.substr(first, last - first);
}

bool is_unquoted_key_char(char ch) {
  return ch != '.' && ch != '[' && ch != ']';
}

std::size_t parse_unsigned_index(const std::string& text, std::size_t begin, std::size_t end) {
  if (begin >= end) {
    throw std::invalid_argument("array index must be non-empty");
  }
  std::size_t value = 0;
  for (std::size_t pos = begin; pos < end; ++pos) {
    const unsigned char ch = static_cast<unsigned char>(text[pos]);
    if (std::isdigit(ch) == 0) {
      throw std::invalid_argument("array index must contain only digits");
    }
    value = (value * 10) + static_cast<std::size_t>(text[pos] - '0');
  }
  return value;
}

std::string parse_quoted_key(const std::string& text, std::size_t* pos) {
  const char quote = text[*pos];
  ++(*pos);
  std::string out;
  while (*pos < text.size()) {
    const char ch = text[*pos];
    ++(*pos);
    if (ch == quote) {
      return out;
    }
    if (ch != '\\') {
      out.push_back(ch);
      continue;
    }
    if (*pos >= text.size()) {
      throw std::invalid_argument("unterminated escape sequence in quoted key");
    }
    const char escaped = text[*pos];
    ++(*pos);
    if (escaped == '\\' || escaped == '"' || escaped == '\'') {
      out.push_back(escaped);
      continue;
    }
    if (escaped == 'n') {
      out.push_back('\n');
      continue;
    }
    if (escaped == 't') {
      out.push_back('\t');
      continue;
    }
    throw std::invalid_argument("unsupported escape sequence in quoted key");
  }
  throw std::invalid_argument("unterminated quoted key");
}

std::vector<DataPickPathSegment> parse_data_pick_path(const std::string& raw_path) {
  const std::string path = trim_string(raw_path);
  std::vector<DataPickPathSegment> segments;
  if (path.empty() || path == "$") {
    return segments;
  }

  std::size_t pos = 0;
  if (path[pos] == '$') {
    ++pos;
    if (pos < path.size() && path[pos] == '.') {
      ++pos;
    }
  }

  while (pos < path.size()) {
    if (path[pos] == '.') {
      ++pos;
      if (pos >= path.size()) {
        throw std::invalid_argument("path cannot end with '.'");
      }
      continue;
    }

    if (path[pos] == '[') {
      ++pos;
      if (pos >= path.size()) {
        throw std::invalid_argument("unterminated bracket path segment");
      }
      if (path[pos] == '"' || path[pos] == '\'') {
        std::string key = parse_quoted_key(path, &pos);
        if (pos >= path.size() || path[pos] != ']') {
          throw std::invalid_argument("quoted key segment must end with ']'");
        }
        ++pos;
        segments.push_back(DataPickPathSegment{DataPickPathSegmentKind::kKey, std::move(key), 0});
        continue;
      }

      const std::size_t index_begin = pos;
      while (pos < path.size() && path[pos] != ']') {
        ++pos;
      }
      if (pos >= path.size()) {
        throw std::invalid_argument("unterminated array index segment");
      }
      const std::size_t index = parse_unsigned_index(path, index_begin, pos);
      ++pos;
      segments.push_back(DataPickPathSegment{DataPickPathSegmentKind::kIndex, "", index});
      continue;
    }

    const std::size_t key_begin = pos;
    while (pos < path.size() && is_unquoted_key_char(path[pos])) {
      ++pos;
    }
    std::string key = trim_string(path.substr(key_begin, pos - key_begin));
    if (key.empty()) {
      throw std::invalid_argument("object key path segment must be non-empty");
    }
    segments.push_back(DataPickPathSegment{DataPickPathSegmentKind::kKey, std::move(key), 0});
  }

  return segments;
}

std::optional<json> pick_json_path(const json& root, const std::vector<DataPickPathSegment>& path) {
  const json* current = &root;
  for (const auto& segment : path) {
    if (segment.kind == DataPickPathSegmentKind::kKey) {
      if (!current->is_object()) {
        return std::nullopt;
      }
      const auto it = current->find(segment.key);
      if (it == current->end()) {
        return std::nullopt;
      }
      current = &(*it);
      continue;
    }

    if (!current->is_array() || segment.index >= current->size()) {
      return std::nullopt;
    }
    current = &((*current)[segment.index]);
  }
  return *current;
}

std::string normalize_value_type(const json& value) {
  std::string text = value.is_string() ? value.get<std::string>() : "any";
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (text == "number" || text == "string" || text == "bool" || text == "boolean" || text == "any") {
    return text == "boolean" ? "bool" : text;
  }
  return "any";
}

std::optional<bool> json_bool(const json& value) {
  if (value.is_boolean())
    return value.get<bool>();
  if (value.is_number_integer())
    return value.get<std::int64_t>() != 0;
  if (value.is_number_unsigned())
    return value.get<std::uint64_t>() != 0;
  if (value.is_string()) {
    std::string text = value.get<std::string>();
    for (char& ch : text) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (text == "1" || text == "true" || text == "yes" || text == "on")
      return true;
    if (text == "0" || text == "false" || text == "no" || text == "off")
      return false;
  }
  return std::nullopt;
}

json json_string_value(const json& value) {
  if (value.is_null())
    return nullptr;
  if (value.is_string())
    return value.get<std::string>();
  if (value.is_boolean())
    return value.get<bool>() ? "true" : "false";
  if (value.is_number())
    return value.dump();
  return value.dump();
}

json coerce_picked_value(const json& value, const std::string& value_type, const json& fallback) {
  if (value_type == "any") {
    return value;
  }
  if (value_type == "number") {
    const auto numeric = json_number(value);
    return numeric.has_value() ? json(numeric.value()) : fallback;
  }
  if (value_type == "bool") {
    const auto boolean = json_bool(value);
    return boolean.has_value() ? json(boolean.value()) : fallback;
  }
  if (value_type == "string") {
    const json string_value = json_string_value(value);
    return string_value.is_null() ? fallback : string_value;
  }
  return value;
}

class DataPickNode final : public OperatorNode, public ComputableNode {
 public:
  DataPickNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"msg"}), data_port_names(node.dataOutPorts, {"out"}),
                     state_names(node.stateFields, {"path", "valueType", "fallback"}),
                     strings_or(node.execInPorts, {"exec"}), strings_or(node.execOutPorts, {"exec"})) {
    path_ = initial_state.value("path", "");
    value_type_ = normalize_value_type(initial_state.value("valueType", "any"));
    fallback_ = initial_state.contains("fallback") ? initial_state["fallback"] : json(nullptr);
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "path")
      return value.is_string() ? value.get<std::string>() : "";
    if (field == "valueType")
      return normalize_value_type(value);
    return value;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "path") {
      path_ = value.is_string() ? value.get<std::string>() : "";
      path_ready_ = false;
      dirty_ = true;
      return;
    }
    if (field == "valueType") {
      value_type_ = normalize_value_type(value);
      dirty_ = true;
      return;
    }
    if (field == "fallback") {
      fallback_ = value;
      dirty_ = true;
    }
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    const std::string port = default_output_port();
    const json value = compute_output(port, exec_id);
    if (!value.is_null()) {
      (void)emit(port, value);
    }
    return exec_out_ports();
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)value;
    (void)meta;
    if (port != default_input_port())
      return;
    const std::string output_port = default_output_port();
    const json output = compute_output(output_port, ts_ms);
    if (!output.is_null()) {
      (void)emit(output_port, output, ts_ms);
    }
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    const std::string output_port = default_output_port();
    if (output_port.empty() || port != output_port)
      return nullptr;
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && last_outputs_.contains(port)) {
      return last_outputs_[port];
    }

    last_outputs_ = json::object();
    try {
      compile_path_if_needed();
      const auto input = pull(default_input_port(), ctx_id);
      const json picked = input.has_value() ? pick_or_fallback(input.value()) : fallback_;
      last_outputs_[output_port] = picked;
      clear_error("data-pick:" + node_id());
    } catch (const std::exception& exc) {
      report_error("DATA_PICK_ERROR", exc.what(), "error", "data-pick:" + node_id(), ctx_id);
      last_outputs_[output_port] = fallback_;
    }

    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return object_value_or_null(last_outputs_, port);
  }

 private:
  std::string default_input_port() const {
    for (const auto& port : data_in_ports()) {
      if (port == "msg")
        return port;
    }
    if (!data_in_ports().empty())
      return data_in_ports().front();
    return "";
  }

  std::string default_output_port() const {
    for (const auto& port : data_out_ports()) {
      if (port == "out")
        return port;
    }
    if (!data_out_ports().empty())
      return data_out_ports().front();
    return "";
  }

  void compile_path_if_needed() {
    if (path_ready_)
      return;
    parsed_path_ = parse_data_pick_path(path_);
    path_ready_ = true;
  }

  json pick_or_fallback(const json& input) const {
    const auto picked = pick_json_path(input, parsed_path_);
    if (!picked.has_value()) {
      return fallback_;
    }
    return coerce_picked_value(picked.value(), value_type_, fallback_);
  }

  std::string path_;
  std::string value_type_ = "any";
  json fallback_ = nullptr;
  std::vector<DataPickPathSegment> parsed_path_;
  json last_outputs_ = json::object();
  std::optional<std::int64_t> last_ctx_id_;
  bool path_ready_ = false;
  bool dirty_ = true;
};

json data_pick_spec() {
  return json{
      {"specKind", "operator"},
      {"schemaVersion", "f8operator/1"},
      {"serviceClass", kServiceClass},
      {"paletteCategory", std::string(kServiceClass) + ".expr"},
      {"operatorClass", "f8.data_pick"},
      {"version", "0.0.1"},
      {"label", "Data Pick"},
      {"description",
       "Pick a value from a JSON-compatible input payload using a small path such as center.y, pos[1], or "
       "[\"weird-key\"].score."},
      {"tags", json::array({"data", "json", "path", "pick", "extract", "transform"})},
      {"dataInPorts", json::array({data_port("msg", "JSON-compatible input payload.", any_schema(), false, true)})},
      {"dataOutPorts", json::array({data_port("out", "Picked value.", any_schema(), false, true)})},
      {"editPolicy",
       json{{"dataInPorts", editable_collection_policy()}, {"dataOutPorts", editable_collection_policy()}}},
      {"stateFields",
       json::array({state_field("path", "Path",
                                "Path to pick. Supports dot keys, zero-based array indexes, and quoted bracket keys.",
                                string_schema(""), "rw", true, true, json{{"kind", "textarea"}}),
                    state_field("valueType", "Value Type", "Coerce picked value before output.",
                                string_enum_schema("any", {"any", "number", "string", "bool"}), "rw", true, true),
                    state_field("fallback", "Fallback", "Value emitted when the path is missing or coercion fails.",
                                json{{"type", "any"}, {"default", nullptr}}, "rw", false, false,
                                json{{"kind", "textarea"}, {"language", "json"}})})}};
}

}  // namespace

void register_data_pick_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(data_pick_spec(), true);
  registry.register_operator_factory(
      kServiceClass, "f8.data_pick",
      [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
        return std::make_unique<DataPickNode>(node_id, node, initial_state);
      },
      true);
}

}  // namespace f8::cppengine
