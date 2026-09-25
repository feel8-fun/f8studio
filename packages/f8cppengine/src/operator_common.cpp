#include "operator_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <stdexcept>

namespace f8::cppengine {

json any_schema() { return json{{"type", "any"}}; }
json number_schema(double default_value) { return json{{"type", "number"}, {"default", default_value}}; }
json number_schema(double default_value, double minimum, double maximum) {
  json out{{"type", "number"}, {"default", default_value}, {"minimum", minimum}};
  if (maximum > minimum) out["maximum"] = maximum;
  return out;
}
json integer_schema(std::int64_t default_value, std::int64_t minimum, std::int64_t maximum) {
  json out{{"type", "integer"}, {"default", default_value}, {"minimum", minimum}};
  if (maximum > minimum) out["maximum"] = maximum;
  return out;
}
json boolean_schema(bool default_value) { return json{{"type", "boolean"}, {"default", default_value}}; }
json string_schema(const std::string& default_value) { return json{{"type", "string"}, {"default", default_value}}; }
json string_enum_schema(const std::string& default_value, std::vector<std::string> values) {
  return json{{"type", "string"}, {"default", default_value}, {"enum", values}};
}
json array_schema(const json& items) { return json{{"type", "array"}, {"items", items}}; }

json data_port(const std::string& name, const std::string& description, const json& schema, bool definition_protected,
               bool show_on_node) {
  return json{{"name", name},
              {"description", description},
              {"valueSchema", schema},
              {"definitionProtected", definition_protected},
              {"showOnNode", show_on_node}};
}

json state_field(const std::string& name, const std::string& label, const std::string& description, const json& schema,
                 const std::string& access, bool value_required, bool show_on_node, const json& control) {
  json out{{"name", name},
           {"label", label},
           {"description", description},
           {"valueSchema", schema},
           {"access", access},
           {"valueRequired", value_required},
           {"showOnNode", show_on_node}};
  if (!control.is_null()) {
    out["control"] = control;
  }
  return out;
}

json editable_collection_policy() {
  return json{{"canAdd", true}, {"canRemove", true}, {"canRename", true}};
}

json editable_script_policy() {
  return json{{"stateFields", editable_collection_policy()},
              {"dataInPorts", editable_collection_policy()},
              {"dataOutPorts", editable_collection_policy()},
              {"execInPorts", editable_collection_policy()},
              {"execOutPorts", editable_collection_policy()}};
}

json object_value_or_null(const json& object, const std::string& key) {
  if (!object.is_object()) return json(nullptr);
  const auto it = object.find(key);
  if (it == object.end()) return json(nullptr);
  return *it;
}

std::vector<std::string> data_port_names(const std::optional<std::vector<f8::cppsdk::generated::F8DataPortSpec>>& ports,
                                         std::vector<std::string> fallback) {
  std::vector<std::string> out;
  for (const auto& port : ports.value_or(std::vector<f8::cppsdk::generated::F8DataPortSpec>{})) {
    out.push_back(port.name);
  }
  return out.empty() ? std::move(fallback) : out;
}

std::vector<std::string> state_names(const std::optional<std::vector<f8::cppsdk::generated::F8StateSpec>>& fields,
                                     std::vector<std::string> fallback) {
  std::vector<std::string> out;
  for (const auto& field : fields.value_or(std::vector<f8::cppsdk::generated::F8StateSpec>{})) {
    out.push_back(field.name);
  }
  return out.empty() ? std::move(fallback) : out;
}

std::vector<std::string> strings_or(const std::optional<std::vector<std::string>>& values,
                                    std::vector<std::string> fallback) {
  if (values.has_value() && !values->empty()) return values.value();
  return fallback;
}

double json_number_or(const json& value, double fallback) {
  if (value.is_number()) return value.get<double>();
  if (value.is_string()) {
    try {
      return std::stod(value.get<std::string>());
    } catch (const std::exception&) {
      return fallback;
    }
  }
  return fallback;
}

std::optional<double> json_number(const json& value) {
  if (value.is_number()) return value.get<double>();
  if (value.is_string()) {
    try {
      return std::stod(value.get<std::string>());
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

bool json_bool_or(const json& value, bool fallback) {
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number_integer()) return value.get<int>() != 0;
  if (value.is_string()) {
    std::string text = value.get<std::string>();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (text == "1" || text == "true" || text == "yes" || text == "on") return true;
    if (text == "0" || text == "false" || text == "no" || text == "off") return false;
  }
  return fallback;
}

double clamp_double(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

int js_round(double value) {
  if (value >= 0.0) return static_cast<int>(std::floor(value + 0.5));
  return -static_cast<int>(std::floor(std::abs(value) + 0.5));
}

std::string json_to_printable(json value, bool strip) {
  std::string text;
  if (value.is_string()) {
    text = value.get<std::string>();
  } else {
    text = value.dump();
  }
  if (!strip) return text;
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), text.end());
  return text;
}

std::vector<double> json_number_sequence(const json& value) {
  std::vector<double> out;
  if (const auto numeric = json_number(value); numeric.has_value()) {
    out.push_back(numeric.value());
    return out;
  }
  if (!value.is_array()) return {};
  for (const auto& item : value) {
    const auto numeric = json_number(item);
    if (!numeric.has_value()) return {};
    out.push_back(numeric.value());
  }
  return out;
}

json format_number_sequence(const std::vector<double>& values) {
  if (values.empty()) return nullptr;
  if (values.size() == 1) return values.front();
  return values;
}

double now_seconds() {
  return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count()) /
         1000000.0;
}

const double kPi = 3.141592653589793238462643383279502884;
const double kTwoPi = 2.0 * kPi;

std::string normalize_curve(const json& value) {
  std::string curve = value.is_string() ? value.get<std::string>() : "";
  std::transform(curve.begin(), curve.end(), curve.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  static const std::vector<std::string> choices{"LINEAR", "SMOOTHSTEP", "SMOOTHERSTEP", "EASE_IN", "EASE_OUT",
                                                "EASE_IN_OUT"};
  for (const auto& choice : choices) {
    if (curve == choice) return curve;
  }
  return "LINEAR";
}

double apply_curve(const std::string& curve, double t) {
  if (curve == "SMOOTHSTEP") return t * t * (3.0 - 2.0 * t);
  if (curve == "SMOOTHERSTEP") return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  if (curve == "EASE_IN") return t * t;
  if (curve == "EASE_OUT") return 1.0 - (1.0 - t) * (1.0 - t);
  if (curve == "EASE_IN_OUT") {
    if (t < 0.5) return 2.0 * t * t;
    return 1.0 - 2.0 * (1.0 - t) * (1.0 - t);
  }
  return t;
}

}  // namespace f8::cppengine
