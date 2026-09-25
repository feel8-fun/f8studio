#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "f8cppsdk/generated/protocol_models.h"

namespace f8::cppengine {

using json = nlohmann::json;

json any_schema();
json number_schema(double default_value = 0.0);
json number_schema(double default_value, double minimum, double maximum = 0.0);
json integer_schema(std::int64_t default_value, std::int64_t minimum = 0, std::int64_t maximum = 0);
json boolean_schema(bool default_value);
json string_schema(const std::string& default_value);
json string_enum_schema(const std::string& default_value, std::vector<std::string> values);
json array_schema(const json& items = any_schema());

json data_port(const std::string& name, const std::string& description, const json& schema,
               bool definition_protected = false,
               bool show_on_node = true);
json state_field(const std::string& name, const std::string& label, const std::string& description, const json& schema,
                 const std::string& access = "rw", bool value_required = true, bool show_on_node = false,
                 const json& control = nullptr);
json editable_collection_policy();
json editable_script_policy();
json object_value_or_null(const json& object, const std::string& key);

std::vector<std::string> data_port_names(const std::optional<std::vector<f8::cppsdk::generated::F8DataPortSpec>>& ports,
                                         std::vector<std::string> fallback);
std::vector<std::string> state_names(const std::optional<std::vector<f8::cppsdk::generated::F8StateSpec>>& fields,
                                     std::vector<std::string> fallback);
std::vector<std::string> strings_or(const std::optional<std::vector<std::string>>& values,
                                    std::vector<std::string> fallback);

double json_number_or(const json& value, double fallback);
std::optional<double> json_number(const json& value);
bool json_bool_or(const json& value, bool fallback);
double clamp_double(double value, double lo, double hi);
int js_round(double value);
std::string json_to_printable(json value, bool strip);
std::vector<double> json_number_sequence(const json& value);
json format_number_sequence(const std::vector<double>& values);
double now_seconds();

extern const double kPi;
extern const double kTwoPi;

std::string normalize_curve(const json& value);
double apply_curve(const std::string& curve, double t);

}  // namespace f8::cppengine
