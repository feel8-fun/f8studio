#include "f8cppsdk/latest_video_frame_transport.h"

#include "f8cppsdk/latest_binary_stream_transport.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

namespace f8::cppsdk {
namespace {

constexpr auto kDecodeFailureWarnDelay = std::chrono::seconds(5);
constexpr auto kDecodeFailureWarnInterval = std::chrono::seconds(10);

void set_error(std::string* error_message, std::string value) {
  if (error_message != nullptr) {
    *error_message = std::move(value);
  }
}

void write_u32_le(std::uint8_t* out, std::size_t offset, std::uint32_t value) {
  out[offset] = static_cast<std::uint8_t>(value & 0xFFu);
  out[offset + 1] = static_cast<std::uint8_t>((value >> 8u) & 0xFFu);
  out[offset + 2] = static_cast<std::uint8_t>((value >> 16u) & 0xFFu);
  out[offset + 3] = static_cast<std::uint8_t>((value >> 24u) & 0xFFu);
}

void write_u64_le(std::uint8_t* out, std::size_t offset, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    out[offset + shift / 8] = static_cast<std::uint8_t>((value >> shift) & 0xFFu);
  }
}

void write_i64_le(std::uint8_t* out, std::size_t offset, std::int64_t value) {
  write_u64_le(out, offset, static_cast<std::uint64_t>(value));
}

bool read_u32_le(const std::uint8_t* data, std::size_t size, std::size_t offset, std::uint32_t& out) {
  if (data == nullptr || offset > size || size - offset < 4) {
    return false;
  }
  out = static_cast<std::uint32_t>(data[offset]) | (static_cast<std::uint32_t>(data[offset + 1]) << 8u) |
        (static_cast<std::uint32_t>(data[offset + 2]) << 16u) |
        (static_cast<std::uint32_t>(data[offset + 3]) << 24u);
  return true;
}

bool read_u64_le(const std::uint8_t* data, std::size_t size, std::size_t offset, std::uint64_t& out) {
  if (data == nullptr || offset > size || size - offset < 8) {
    return false;
  }
  out = 0;
  for (unsigned index = 0; index < 8; ++index) {
    out |= static_cast<std::uint64_t>(data[offset + index]) << (index * 8u);
  }
  return true;
}

bool read_i64_le(const std::uint8_t* data, std::size_t size, std::size_t offset, std::int64_t& out) {
  std::uint64_t value = 0;
  if (!read_u64_le(data, size, offset, value)) {
    return false;
  }
  out = static_cast<std::int64_t>(value);
  return true;
}

bool validate_zenoh_video_frame(const VideoFrameView& frame, std::size_t& frame_bytes, std::string* error_message) {
  frame_bytes = 0;
  if (frame.width == 0 || frame.height == 0 || frame.pitch == 0) {
    set_error(error_message, "width, height, and pitch must be positive");
    return false;
  }
  if (frame.format == 0) {
    set_error(error_message, "format must be positive");
    return false;
  }
  if (frame.frame_id == 0) {
    set_error(error_message, "frame_id must be positive");
    return false;
  }
  if (frame.payload == nullptr) {
    set_error(error_message, "payload must be non-null");
    return false;
  }
  frame_bytes = static_cast<std::size_t>(frame.pitch) * static_cast<std::size_t>(frame.height);
  if (frame_bytes == 0 || frame.payload_bytes < frame_bytes) {
    set_error(error_message, "payload is smaller than pitch * height");
    return false;
  }
  if (frame_bytes > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    set_error(error_message, "payload is too large for zenoh video frame schema v2");
    return false;
  }
  return true;
}

void write_zenoh_video_frame_unchecked(const VideoFrameView& frame, std::size_t frame_bytes, std::uint8_t* out) {
  write_u32_le(out, 0, kZenohVideoFrameMagic);
  write_u32_le(out, 4, kZenohVideoFrameSchemaVersion);
  write_u32_le(out, 8, kZenohVideoFrameHeaderBytes);
  write_u32_le(out, 12, frame.width);
  write_u32_le(out, 16, frame.height);
  write_u32_le(out, 20, frame.pitch);
  write_u32_le(out, 24, frame.format);
  write_u32_le(out, 28, static_cast<std::uint32_t>(frame_bytes));
  write_u64_le(out, 32, frame.frame_id);
  write_i64_le(out, 40, frame.ts_ms);
  write_u64_le(out, 48, frame.stream_epoch_high);
  write_u64_le(out, 56, frame.stream_epoch_low);
  const auto* payload = reinterpret_cast<const std::uint8_t*>(frame.payload);
  std::memcpy(out + kZenohVideoFrameHeaderBytes, payload, frame_bytes);
}

bool decode_zenoh_video_frame_from_buffer(const std::uint8_t* raw, std::size_t raw_size, LatestVideoFrame& out,
                                          std::string* error_message) {
  out = LatestVideoFrame{};
  if (raw == nullptr || raw_size < kZenohVideoFrameHeaderBytes) {
    set_error(error_message, "payload is smaller than zenoh video frame header");
    return false;
  }

  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t header_bytes = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t pitch = 0;
  std::uint32_t format = 0;
  std::uint32_t payload_bytes = 0;
  std::uint64_t frame_id = 0;
  std::int64_t ts_ms = 0;
  std::uint64_t stream_epoch_high = 0;
  std::uint64_t stream_epoch_low = 0;
  if (!read_u32_le(raw, raw_size, 0, magic) || !read_u32_le(raw, raw_size, 4, version) ||
      !read_u32_le(raw, raw_size, 8, header_bytes) || !read_u32_le(raw, raw_size, 12, width) ||
      !read_u32_le(raw, raw_size, 16, height) || !read_u32_le(raw, raw_size, 20, pitch) ||
      !read_u32_le(raw, raw_size, 24, format) || !read_u32_le(raw, raw_size, 28, payload_bytes) ||
      !read_u64_le(raw, raw_size, 32, frame_id) || !read_i64_le(raw, raw_size, 40, ts_ms) ||
      !read_u64_le(raw, raw_size, 48, stream_epoch_high) ||
      !read_u64_le(raw, raw_size, 56, stream_epoch_low)) {
    set_error(error_message, "payload header is truncated");
    return false;
  }
  if (magic != kZenohVideoFrameMagic || version != kZenohVideoFrameSchemaVersion) {
    set_error(error_message, "unsupported zenoh video frame schema");
    return false;
  }
  if (header_bytes < kZenohVideoFrameHeaderBytes) {
    set_error(error_message, "invalid zenoh video frame header size");
    return false;
  }
  if (width == 0 || height == 0 || pitch == 0 || format == 0 || frame_id == 0) {
    set_error(error_message, "invalid zenoh video frame metadata");
    return false;
  }
  const std::size_t expected_payload_bytes = static_cast<std::size_t>(pitch) * static_cast<std::size_t>(height);
  if (payload_bytes != expected_payload_bytes) {
    set_error(error_message, "zenoh video frame payload size does not match pitch * height");
    return false;
  }
  if (static_cast<std::size_t>(header_bytes) > raw_size ||
      raw_size - static_cast<std::size_t>(header_bytes) < static_cast<std::size_t>(payload_bytes)) {
    set_error(error_message, "zenoh video frame payload is truncated");
    return false;
  }

  out.width = width;
  out.height = height;
  out.pitch = pitch;
  out.format = format;
  out.frame_id = frame_id;
  out.ts_ms = ts_ms;
  out.stream_epoch_high = stream_epoch_high;
  out.stream_epoch_low = stream_epoch_low;
  out.payload.resize(payload_bytes);
  std::memcpy(out.payload.data(), raw + header_bytes, payload_bytes);
  return true;
}

}  // namespace

bool encode_zenoh_video_frame(const VideoFrameView& frame, RuntimeBytes& out, std::string* error_message) {
  out.clear();
  std::size_t frame_bytes = 0;
  if (!validate_zenoh_video_frame(frame, frame_bytes, error_message)) {
    return false;
  }

  out.resize(static_cast<std::size_t>(kZenohVideoFrameHeaderBytes) + frame_bytes);
  write_zenoh_video_frame_unchecked(frame, frame_bytes, out.data());
  return true;
}

bool decode_zenoh_video_frame(const RuntimeBytes& raw, LatestVideoFrame& out, std::string* error_message) {
  return decode_zenoh_video_frame_from_buffer(raw.data(), raw.size(), out, error_message);
}

class ZenohLatestVideoFramePublisher::Impl final {
 public:
  Impl() : publisher_("video") {}

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr) {
    std::random_device random;
    stream_epoch_high_ = (static_cast<std::uint64_t>(random()) << 32u) | random();
    stream_epoch_low_ = (static_cast<std::uint64_t>(random()) << 32u) | random();
    if (stream_epoch_high_ == 0 && stream_epoch_low_ == 0) {
      stream_epoch_low_ = 1;
    }
    return publisher_.open(config, key_expr);
  }

  void close() {
    publisher_.close();
  }

  bool publish_frame(const VideoFrameView& frame) {
    VideoFrameView frame_with_epoch = frame;
    frame_with_epoch.stream_epoch_high = stream_epoch_high_;
    frame_with_epoch.stream_epoch_low = stream_epoch_low_;
    std::size_t frame_bytes = 0;
    std::string error;
    if (!validate_zenoh_video_frame(frame_with_epoch, frame_bytes, &error)) {
      spdlog::error("zenoh video frame encode failed key={}: {}", publisher_.key_expr(), error);
      return false;
    }
    const std::size_t encoded_bytes = static_cast<std::size_t>(kZenohVideoFrameHeaderBytes) + frame_bytes;
    return publisher_.publish_payload(
        encoded_bytes,
        [frame_with_epoch, frame_bytes](std::uint8_t* out, std::size_t size, std::string* error_message) {
          if (out == nullptr || size < static_cast<std::size_t>(kZenohVideoFrameHeaderBytes) + frame_bytes) {
            set_error(error_message, "output buffer is smaller than encoded video frame");
            return false;
          }
          write_zenoh_video_frame_unchecked(frame_with_epoch, frame_bytes, out);
          return true;
        });
  }

  bool valid() const {
    return publisher_.valid();
  }

  std::string key_expr() const {
    return publisher_.key_expr();
  }

 private:
  std::uint64_t stream_epoch_high_ = 0;
  std::uint64_t stream_epoch_low_ = 0;
  ZenohLatestBinaryStreamPublisher publisher_;
};

class ZenohLatestVideoFrameSubscriber::Impl final {
 public:
  Impl() : subscriber_("video") {}

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr) {
    reset_decode_failure_tracking();
    return subscriber_.open(config, key_expr);
  }

  void close() {
    subscriber_.close();
    reset_decode_failure_tracking();
  }

  std::optional<LatestVideoFrame> poll_latest() {
    std::optional<RuntimeBytes> raw = subscriber_.poll_latest();
    if (!raw.has_value()) {
      return std::nullopt;
    }
    return decode_latest(*raw);
  }

  std::optional<LatestVideoFrame> wait_latest(std::chrono::milliseconds timeout) {
    std::optional<RuntimeBytes> raw = subscriber_.wait_latest(timeout);
    if (!raw.has_value()) {
      return std::nullopt;
    }
    return decode_latest(*raw);
  }

  bool valid() const {
    return subscriber_.valid();
  }

  std::string key_expr() const {
    return subscriber_.key_expr();
  }

 private:
  std::optional<LatestVideoFrame> decode_latest(const RuntimeBytes& raw) {
    LatestVideoFrame frame;
    std::string error;
    if (!decode_zenoh_video_frame(raw, frame, &error)) {
      record_decode_failure(error);
      return std::nullopt;
    }
    reset_decode_failure_tracking();
    return frame;
  }

  void reset_decode_failure_tracking() {
    decode_failure_count_ = 0;
    first_decode_failure_at_ = std::chrono::steady_clock::time_point{};
    last_decode_failure_warn_at_ = std::chrono::steady_clock::time_point{};
  }

  void record_decode_failure(const std::string& error) {
    const auto now = std::chrono::steady_clock::now();
    if (decode_failure_count_ == 0) {
      first_decode_failure_at_ = now;
    }
    ++decode_failure_count_;

    const bool warning_due =
        first_decode_failure_at_ != std::chrono::steady_clock::time_point{} &&
        (now - first_decode_failure_at_) >= kDecodeFailureWarnDelay &&
        (last_decode_failure_warn_at_ == std::chrono::steady_clock::time_point{} ||
         (now - last_decode_failure_warn_at_) >= kDecodeFailureWarnInterval);
    if (!warning_due) {
      return;
    }

    last_decode_failure_warn_at_ = now;
    spdlog::warn("zenoh video frame decode skipped invalid samples key={} count={} last={}", subscriber_.key_expr(),
                 decode_failure_count_, error);
  }

  ZenohLatestBinaryStreamSubscriber subscriber_;
  std::uint64_t decode_failure_count_ = 0;
  std::chrono::steady_clock::time_point first_decode_failure_at_{};
  std::chrono::steady_clock::time_point last_decode_failure_warn_at_{};
};

ZenohLatestVideoFramePublisher::ZenohLatestVideoFramePublisher() : impl_(std::make_unique<Impl>()) {}
ZenohLatestVideoFramePublisher::~ZenohLatestVideoFramePublisher() {
  close();
}

bool ZenohLatestVideoFramePublisher::open(const RuntimeBackendConfig& config, const std::string& key_expr) {
  return impl_->open(config, key_expr);
}

void ZenohLatestVideoFramePublisher::close() {
  impl_->close();
}

bool ZenohLatestVideoFramePublisher::publish_frame(const VideoFrameView& frame) {
  return impl_->publish_frame(frame);
}

bool ZenohLatestVideoFramePublisher::valid() const {
  return impl_->valid();
}

std::string ZenohLatestVideoFramePublisher::key_expr() const {
  return impl_->key_expr();
}

ZenohLatestVideoFrameSubscriber::ZenohLatestVideoFrameSubscriber() : impl_(std::make_unique<Impl>()) {}
ZenohLatestVideoFrameSubscriber::~ZenohLatestVideoFrameSubscriber() {
  close();
}

bool ZenohLatestVideoFrameSubscriber::open(const RuntimeBackendConfig& config, const std::string& key_expr) {
  return impl_->open(config, key_expr);
}

void ZenohLatestVideoFrameSubscriber::close() {
  impl_->close();
}

std::optional<LatestVideoFrame> ZenohLatestVideoFrameSubscriber::poll_latest() {
  return impl_->poll_latest();
}

std::optional<LatestVideoFrame> ZenohLatestVideoFrameSubscriber::wait_latest(std::chrono::milliseconds timeout) {
  return impl_->wait_latest(timeout);
}

bool ZenohLatestVideoFrameSubscriber::valid() const {
  return impl_->valid();
}

std::string ZenohLatestVideoFrameSubscriber::key_expr() const {
  return impl_->key_expr();
}

}  // namespace f8::cppsdk
