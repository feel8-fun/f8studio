#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace f8::cppsdk::describe {

nlohmann::json schema_string();
nlohmann::json schema_string_enum(std::initializer_list<const char*> items);
nlohmann::json schema_string_enum(const std::vector<std::string>& values, const std::string& default_value = {});
nlohmann::json schema_number();
nlohmann::json schema_number(double default_value, double minimum, double maximum);
nlohmann::json schema_integer();
nlohmann::json schema_integer(std::int64_t default_value, std::int64_t minimum, std::int64_t maximum);
nlohmann::json schema_boolean();
nlohmann::json schema_object(
    const nlohmann::json& props,
    const nlohmann::json& required = nlohmann::json::array());
nlohmann::json schema_array(const nlohmann::json& item_schema);
nlohmann::json schema_video_frame_metadata();
nlohmann::json schema_video_frame();
nlohmann::json schema_audio_chunk_metadata();
nlohmann::json schema_audio_chunk();
nlohmann::json data_stream(
    std::string delivery = "fifo",
    std::string reliability = "best_effort",
    std::string congestion = "drop",
    std::string priority = "data");
nlohmann::json data_port(
    std::string name,
    const nlohmann::json& value_schema,
    std::string payload_kind = "json",
    std::string delivery = "fifo",
    std::string description = {},
    bool definition_protected = true,
    bool show_on_node = true,
    const nlohmann::json& metadata_schema = nlohmann::json(),
    const std::vector<std::string>& formats = {},
    std::string reliability = "best_effort",
    std::string congestion = "drop",
    std::string priority = "data",
    std::uint32_t payload_schema_version = 1);
nlohmann::json video_frame_port(std::string name, std::string description = {}, bool definition_protected = true);
nlohmann::json audio_chunk_port(std::string name, std::string description = {}, bool definition_protected = true);
nlohmann::json state_field(
    std::string name,
    const nlohmann::json& value_schema,
    std::string access,
    std::string label = {},
    std::string description = {},
    bool show_on_node = false,
    nlohmann::json control = nullptr,
    bool redact_on_publish = false);

}  // namespace f8::cppsdk::describe
