#pragma once

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/latest_video_frame_transport.h"

namespace f8::cvkit::service_runtime {

using json = nlohmann::json;

struct CvProcessMetrics {
  std::uint64_t observed_frames = 0;
  std::uint64_t processed_frames = 0;
  std::uint64_t dropped_frames = 0;
  std::uint64_t failed_frames = 0;
  double last_process_ms = 0.0;
  double avg_process_ms = 0.0;
  double last_latency_ms = 0.0;
  double avg_latency_ms = 0.0;
  double process_fps = 0.0;
  std::uint64_t last_points_per_frame = 0;
  std::uint64_t last_vectors_per_frame = 0;
};

struct FrameBufferValidationResult {
  bool ok = false;
  std::size_t row_bytes = 0;
  const char* reason = "unknown frame buffer validation failure";
};

inline FrameBufferValidationResult validate_frame_buffer(std::uint32_t actual_format,
                                                         std::uint32_t width,
                                                         std::uint32_t height,
                                                         std::uint32_t pitch,
                                                         std::size_t payload_size,
                                                         std::uint32_t expected_format,
                                                         std::size_t bytes_per_pixel) {
  FrameBufferValidationResult result;
  if (actual_format != expected_format) {
    result.reason = "unexpected frame format";
    return result;
  }
  if (width == 0 || height == 0 || pitch == 0) {
    result.reason = "invalid frame dimensions";
    return result;
  }
  if (width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      pitch > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    result.reason = "frame dimensions exceed OpenCV limits";
    return result;
  }
  if (bytes_per_pixel == 0) {
    result.reason = "invalid expected bytes per pixel";
    return result;
  }

  const std::size_t frame_width = static_cast<std::size_t>(width);
  if (frame_width > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
    result.reason = "frame row byte count overflows";
    return result;
  }
  const std::size_t minimum_row_bytes = frame_width * bytes_per_pixel;
  const std::size_t row_bytes = static_cast<std::size_t>(pitch);
  if (row_bytes < minimum_row_bytes) {
    result.reason = "frame pitch is smaller than width * bytes_per_pixel";
    return result;
  }

  const std::size_t frame_height = static_cast<std::size_t>(height);
  if (frame_height > 0 && row_bytes > std::numeric_limits<std::size_t>::max() / frame_height) {
    result.reason = "frame payload byte count overflows";
    return result;
  }
  const std::size_t required_payload_bytes = row_bytes * frame_height;
  if (payload_size < required_payload_bytes) {
    result.reason = "frame payload is smaller than pitch * height";
    return result;
  }

  result.ok = true;
  result.row_bytes = row_bytes;
  result.reason = "";
  return result;
}

inline FrameBufferValidationResult validate_latest_video_frame(const f8::cppsdk::LatestVideoFrame& frame,
                                                               std::uint32_t expected_format,
                                                               std::size_t bytes_per_pixel) {
  return validate_frame_buffer(frame.format, frame.width, frame.height, frame.pitch, frame.payload.size(),
                               expected_format, bytes_per_pixel);
}

inline double latency_ms_from_timestamps(std::int64_t end_ts_ms, std::int64_t source_ts_ms) {
  if (end_ts_ms <= 0 || source_ts_ms <= 0 || end_ts_ms < source_ts_ms) {
    return 0.0;
  }
  return static_cast<double>(end_ts_ms - source_ts_ms);
}

inline std::string trim_copy(std::string value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.erase(value.begin());
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.pop_back();
  }
  return value;
}

inline std::string to_lower_ascii_copy(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

inline bool parse_int_text(const std::string& raw, int& out) {
  const std::string text = trim_copy(raw);
  if (text.empty()) {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  const long long parsed = std::strtoll(text.c_str(), &end, 10);
  if (end == text.c_str() || end == nullptr || *end != '\0' || errno == ERANGE) {
    return false;
  }
  if (parsed < static_cast<long long>(std::numeric_limits<int>::min()) ||
      parsed > static_cast<long long>(std::numeric_limits<int>::max())) {
    return false;
  }

  out = static_cast<int>(parsed);
  return true;
}

inline bool parse_double_text(const std::string& raw, double& out) {
  const std::string text = trim_copy(raw);
  if (text.empty()) {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || end == nullptr || *end != '\0' || errno == ERANGE || !std::isfinite(parsed)) {
    return false;
  }

  out = parsed;
  return true;
}

inline bool parse_json_int(const json& value, int& out) {
  if (value.is_number_integer()) {
    const long long v = value.get<long long>();
    if (v < static_cast<long long>(std::numeric_limits<int>::min()) ||
        v > static_cast<long long>(std::numeric_limits<int>::max())) {
      return false;
    }
    out = static_cast<int>(v);
    return true;
  }
  if (value.is_number_unsigned()) {
    const unsigned long long v = value.get<unsigned long long>();
    if (v > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
      return false;
    }
    out = static_cast<int>(v);
    return true;
  }
  if (value.is_number_float()) {
    const double v = value.get<double>();
    if (!std::isfinite(v)) {
      return false;
    }
    const double rounded = std::lround(v);
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
      return false;
    }
    out = static_cast<int>(rounded);
    return true;
  }
  if (value.is_string()) {
    return parse_int_text(value.get<std::string>(), out);
  }
  return false;
}

inline bool parse_json_double(const json& value, double& out) {
  if (value.is_number_float()) {
    const double v = value.get<double>();
    if (!std::isfinite(v)) {
      return false;
    }
    out = v;
    return true;
  }
  if (value.is_number_integer()) {
    out = static_cast<double>(value.get<long long>());
    return true;
  }
  if (value.is_number_unsigned()) {
    out = static_cast<double>(value.get<unsigned long long>());
    return true;
  }
  if (value.is_string()) {
    return parse_double_text(value.get<std::string>(), out);
  }
  return false;
}

inline std::string monitor_error_message_from_json(const json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_null()) {
    return {};
  }
  return value.dump();
}

inline void publish_error_if_changed(std::mutex& state_mu,
                                     std::unordered_map<std::string, json>& published_state,
                                     f8::cppsdk::ServiceBus* bus,
                                     const std::string& service_id,
                                     const json& value,
                                     const std::string& source,
                                     const json& meta) {
  (void)meta;
  const std::string message = monitor_error_message_from_json(value);
  const json cached_message = message;
  std::lock_guard<std::mutex> lock(state_mu);
  const std::string cache_key = "$monitor.currentError";
  const auto it = published_state.find(cache_key);
  if (it != published_state.end() && it->second == cached_message) {
    return;
  }

  published_state[cache_key] = cached_message;
  if (bus == nullptr) {
    return;
  }
  if (message.empty()) {
    bus->clear_error(service_id);
  } else {
    bus->report_error(service_id, "CVKIT_SERVICE_ERROR", message, "error", service_id + ":" + source + ":" + message);
  }
}

inline void publish_state_if_changed(std::mutex& state_mu,
                                     std::unordered_map<std::string, json>& published_state,
                                     f8::cppsdk::ServiceBus* bus,
                                     const std::string& service_id,
                                     const std::string& field,
                                     const json& value,
                                     const std::string& source,
                                     const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu);
  const auto it = published_state.find(field);
  if (it != published_state.end() && it->second == value) {
    return;
  }

  published_state[field] = value;
  if (bus != nullptr) {
    (void)bus->publish_state(service_id, field, value, source, meta);
  }
}

inline void publish_cv_process_metrics(f8::cppsdk::ServiceBus* bus, const CvProcessMetrics& metrics) {
  if (bus != nullptr) {
    bus->record_monitor_processed("cv_process");
    bus->record_monitor_timing("cv_process", metrics.last_process_ms, metrics.last_latency_ms);
  }
}

}  // namespace f8::cvkit::service_runtime
