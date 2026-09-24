#include "f8cppsdk/describe_schema.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace f8::cppsdk::describe {

using json = nlohmann::json;

json schema_string() {
  return json{{"type", "string"}};
}

json schema_string_enum(std::initializer_list<const char*> items) {
  json s = schema_string();
  s["enum"] = json::array();
  for (const char* item : items) {
    if (item != nullptr && *item != '\0') {
      s["enum"].push_back(item);
    }
  }
  return s;
}

json schema_string_enum(const std::vector<std::string>& values, const std::string& default_value) {
  json s = schema_string();
  s["enum"] = json::array();
  for (const std::string& value : values) {
    if (!value.empty()) {
      s["enum"].push_back(value);
    }
  }
  if (!default_value.empty()) {
    s["default"] = default_value;
  }
  return s;
}

json schema_number() {
  return json{{"type", "number"}};
}

json schema_number(double default_value, double minimum, double maximum) {
  json s{{"type", "number"}};
  s["default"] = default_value;
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
}

json schema_integer() {
  return json{{"type", "integer"}};
}

json schema_integer(std::int64_t default_value, std::int64_t minimum, std::int64_t maximum) {
  json s{{"type", "integer"}};
  s["default"] = default_value;
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
}

json schema_boolean() {
  return json{{"type", "boolean"}};
}

json schema_object(const json& props, const json& required) {
  json obj;
  obj["type"] = "object";
  obj["properties"] = props;
  if (required.is_array()) {
    obj["required"] = required;
  }
  obj["additionalProperties"] = false;
  return obj;
}

json schema_array(const json& item_schema) {
  json arr;
  arr["type"] = "array";
  arr["items"] = item_schema;
  return arr;
}

json schema_video_frame_metadata() {
  json obj = schema_object(
      json{{"schemaVersion", schema_integer(2, 2, 2)},
           {"format", schema_string_enum({"bgra32", "bgr24", "flow2_f16", "scalar1_f32"})},
           {"width", schema_integer()},
           {"height", schema_integer()},
           {"pitch", schema_integer()},
           {"frameId", schema_integer()},
           {"tsMs", schema_integer()},
           {"streamEpoch", schema_string()}},
      json::array({"schemaVersion", "format", "width", "height", "pitch", "frameId", "tsMs", "streamEpoch"}));
  obj["title"] = "F8 Video Frame Stream Metadata";
  obj["description"] =
      "Decoded metadata for a video_frame data stream. Frame bytes are carried by the runtime stream envelope, not by "
      "this JSON object.";
  return obj;
}

json schema_video_frame() {
  return schema_video_frame_metadata();
}

json schema_audio_chunk_metadata() {
  json obj = schema_object(
      json{{"schemaVersion", schema_integer(1, 1, 1)},
           {"format", schema_string_enum({"f32le"})},
           {"sampleRate", schema_integer()},
           {"channels", schema_integer()},
           {"frames", schema_integer()},
           {"bytesPerFrame", schema_integer()},
           {"seq", schema_integer()},
           {"frameIndex", schema_integer()},
           {"tsMs", schema_integer()}},
      json::array({"schemaVersion", "format", "sampleRate", "channels", "frames", "bytesPerFrame", "seq",
                   "frameIndex", "tsMs"}));
  obj["title"] = "F8 Audio Chunk Stream Metadata";
  obj["description"] =
      "Decoded metadata for an audio_chunk data stream. PCM bytes are carried by the runtime stream envelope, not by "
      "this JSON object.";
  return obj;
}

json schema_audio_chunk() {
  return schema_audio_chunk_metadata();
}

json data_stream(std::string delivery, std::string reliability, std::string congestion, std::string priority) {
  json stream;
  stream["delivery"] = std::move(delivery);
  stream["reliability"] = std::move(reliability);
  stream["congestion"] = std::move(congestion);
  stream["priority"] = std::move(priority);
  return stream;
}

json data_port(std::string name, const json& value_schema, std::string payload_kind, std::string delivery,
               std::string description, bool required, bool show_on_node, const json& metadata_schema,
               const std::vector<std::string>& formats, std::string reliability, std::string congestion,
               std::string priority, std::uint32_t payload_schema_version) {
  json payload;
  payload["kind"] = payload_kind;
  payload["schemaVersion"] = payload_schema_version;
  payload["formats"] = formats;
  if (payload_kind == "json") {
    payload["valueSchema"] = value_schema;
  } else {
    payload["metadataSchema"] = metadata_schema.is_null() ? value_schema : metadata_schema;
  }

  json port;
  port["name"] = std::move(name);
  port["valueSchema"] = value_schema;
  port["payload"] = std::move(payload);
  port["stream"] = data_stream(delivery, std::move(reliability), std::move(congestion), std::move(priority));
  port["payloadKind"] = payload_kind;
  port["delivery"] = std::move(delivery);
  port["required"] = required;
  port["showOnNode"] = show_on_node;
  if (!description.empty()) {
    port["description"] = std::move(description);
  }
  return port;
}

json video_frame_port(std::string name, std::string description, bool required) {
  const json metadata = schema_video_frame_metadata();
  return data_port(std::move(name), metadata, "video_frame", "latest", std::move(description), required, true, metadata,
                   {"bgra32", "bgr24", "flow2_f16", "scalar1_f32"}, "best_effort", "drop", "real_time", 2);
}

json audio_chunk_port(std::string name, std::string description, bool required) {
  const json metadata = schema_audio_chunk_metadata();
  return data_port(std::move(name), metadata, "audio_chunk", "latest", std::move(description), required, true, metadata,
                   {"f32le"}, "best_effort", "drop", "real_time", 1);
}

json state_field(std::string name, const json& value_schema, std::string access, std::string label,
                 std::string description, bool show_on_node, std::string ui_control, bool redact_on_publish) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  sf["required"] = true;
  if (!label.empty()) {
    sf["label"] = std::move(label);
  }
  if (!description.empty()) {
    sf["description"] = std::move(description);
  }
  if (show_on_node) {
    sf["showOnNode"] = true;
  }
  if (!ui_control.empty()) {
    sf["uiControl"] = std::move(ui_control);
  }
  if (redact_on_publish) {
    sf["redactOnPublish"] = true;
  }
  return sf;
}

}  // namespace f8::cppsdk::describe
