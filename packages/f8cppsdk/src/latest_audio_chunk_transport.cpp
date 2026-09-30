#include "f8cppsdk/latest_audio_chunk_transport.h"

#include "f8cppsdk/latest_binary_stream_transport.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

namespace f8::cppsdk {
namespace {

void set_error(std::string* error_message, std::string value) {
  if (error_message != nullptr) {
    *error_message = std::move(value);
  }
}

}  // namespace

bool encode_zenoh_audio_chunk(const AudioChunkView& chunk, RuntimeBytes& out, std::string* error_message) {
  out.clear();
  if (chunk.sample_rate == 0 || chunk.channels == 0 || chunk.frames == 0 || chunk.bytes_per_frame == 0) {
    set_error(error_message, "sample_rate, channels, frames, and bytes_per_frame must be positive");
    return false;
  }
  if (chunk.format == 0) {
    set_error(error_message, "format must be positive");
    return false;
  }
  if (chunk.seq == 0) {
    set_error(error_message, "seq must be positive");
    return false;
  }
  if (chunk.payload == nullptr) {
    set_error(error_message, "payload must be non-null");
    return false;
  }
  const std::size_t expected_payload_bytes =
      static_cast<std::size_t>(chunk.frames) * static_cast<std::size_t>(chunk.bytes_per_frame);
  if (expected_payload_bytes == 0 || chunk.payload_bytes < expected_payload_bytes) {
    set_error(error_message, "payload is smaller than frames * bytes_per_frame");
    return false;
  }
  if (expected_payload_bytes > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    set_error(error_message, "payload is too large for zenoh audio chunk schema v1");
    return false;
  }

  out.reserve(static_cast<std::size_t>(kZenohAudioChunkHeaderBytes) + expected_payload_bytes);
  out.resize(kZenohAudioChunkHeaderBytes);
  const AudioChunkHeader header{
    kZenohAudioChunkMagic, kZenohAudioChunkSchemaVersion, kZenohAudioChunkHeaderBytes, chunk.sample_rate, chunk.channels, chunk.format, chunk.frames, chunk.bytes_per_frame, static_cast<std::uint32_t>(expected_payload_bytes), chunk.seq, chunk.frame_index, chunk.ts_ms
  };
  header.encode(out.data());
  const auto* begin = reinterpret_cast<const std::uint8_t*>(chunk.payload);
  out.insert(out.end(), begin, begin + expected_payload_bytes);
  return true;
}

bool decode_zenoh_audio_chunk(const RuntimeBytes& raw, LatestAudioChunk& out, std::string* error_message) {
  out = LatestAudioChunk{};
  if (raw.size() < kZenohAudioChunkHeaderBytes) {
    set_error(error_message, "payload is smaller than zenoh audio chunk header");
    return false;
  }

  AudioChunkHeader header;
  if (!AudioChunkHeader::decode(raw.data(), raw.size(), header)) {
    set_error(error_message, "payload header is truncated");
    return false;
  }
  const auto magic = header.magic;
  const auto version = header.version;
  const auto header_bytes = header.header_bytes;
  const auto sample_rate = header.sample_rate;
  const auto channels = header.channels;
  const auto format = header.fmt;
  const auto frames = header.frames;
  const auto bytes_per_frame = header.bytes_per_frame;
  const auto payload_bytes = header.payload_bytes;
  const auto seq = header.seq;
  const auto frame_index = header.frame_index;
  const auto ts_ms = header.ts_ms;
  if (magic != kZenohAudioChunkMagic || version != kZenohAudioChunkSchemaVersion) {
    set_error(error_message, "unsupported zenoh audio chunk schema");
    return false;
  }
  if (header_bytes < kZenohAudioChunkHeaderBytes) {
    set_error(error_message, "invalid zenoh audio chunk header size");
    return false;
  }
  if (sample_rate == 0 || channels == 0 || format == 0 || frames == 0 || bytes_per_frame == 0 || seq == 0) {
    set_error(error_message, "invalid zenoh audio chunk metadata");
    return false;
  }
  const std::size_t expected_payload_bytes =
      static_cast<std::size_t>(frames) * static_cast<std::size_t>(bytes_per_frame);
  if (payload_bytes != expected_payload_bytes) {
    set_error(error_message, "zenoh audio chunk payload size does not match frames * bytes_per_frame");
    return false;
  }
  if (static_cast<std::size_t>(header_bytes) > raw.size() ||
      raw.size() - static_cast<std::size_t>(header_bytes) < static_cast<std::size_t>(payload_bytes)) {
    set_error(error_message, "zenoh audio chunk payload is truncated");
    return false;
  }

  out.sample_rate = sample_rate;
  out.channels = channels;
  out.format = format;
  out.frames = frames;
  out.bytes_per_frame = bytes_per_frame;
  out.seq = seq;
  out.frame_index = frame_index;
  out.ts_ms = ts_ms;
  out.payload.resize(payload_bytes);
  std::memcpy(out.payload.data(), raw.data() + header_bytes, payload_bytes);
  return true;
}

class ZenohLatestAudioChunkPublisher::Impl final {
 public:
  Impl() : publisher_("audio") {}

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr) {
    return publisher_.open(config, key_expr);
  }

  void close() {
    publisher_.close();
  }

  bool publish_chunk(const AudioChunkView& chunk) {
    RuntimeBytes encoded;
    std::string error;
    if (!encode_zenoh_audio_chunk(chunk, encoded, &error)) {
      report_publish_failure("encode failed: " + error);
      return false;
    }
    const bool ok = publisher_.publish_bytes(encoded);
    publish_failure_reported_ = !ok;
    return ok;
  }

  bool valid() const {
    return publisher_.valid();
  }

  std::string key_expr() const {
    return publisher_.key_expr();
  }

 private:
  void report_publish_failure(const std::string& message) {
    if (publish_failure_reported_) {
      return;
    }
    publish_failure_reported_ = true;
    spdlog::error("zenoh audio publish failed key={}: {}", publisher_.key_expr(), message);
  }

  ZenohLatestBinaryStreamPublisher publisher_;
  bool publish_failure_reported_ = false;
};

class ZenohLatestAudioChunkSubscriber::Impl final {
 public:
  Impl() : subscriber_("audio") {}

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr) {
    decode_failure_reported_ = false;
    return subscriber_.open(config, key_expr);
  }

  void close() {
    subscriber_.close();
    decode_failure_reported_ = false;
  }

  std::optional<LatestAudioChunk> poll_latest() {
    std::optional<RuntimeBytes> raw = subscriber_.poll_latest();
    if (!raw.has_value()) {
      return std::nullopt;
    }
    return decode_latest(*raw);
  }

  std::optional<LatestAudioChunk> wait_latest(std::chrono::milliseconds timeout) {
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
  std::optional<LatestAudioChunk> decode_latest(const RuntimeBytes& raw) {
    LatestAudioChunk chunk;
    std::string error;
    if (!decode_zenoh_audio_chunk(raw, chunk, &error)) {
      if (!decode_failure_reported_) {
        decode_failure_reported_ = true;
        spdlog::error("zenoh audio chunk decode failed key={}: {}", subscriber_.key_expr(), error);
      }
      return std::nullopt;
    }
    decode_failure_reported_ = false;
    return chunk;
  }

  ZenohLatestBinaryStreamSubscriber subscriber_;
  bool decode_failure_reported_ = false;
};

ZenohLatestAudioChunkPublisher::ZenohLatestAudioChunkPublisher() : impl_(std::make_unique<Impl>()) {}
ZenohLatestAudioChunkPublisher::~ZenohLatestAudioChunkPublisher() {
  close();
}

bool ZenohLatestAudioChunkPublisher::open(const RuntimeBackendConfig& config, const std::string& key_expr) {
  return impl_->open(config, key_expr);
}

void ZenohLatestAudioChunkPublisher::close() {
  impl_->close();
}

bool ZenohLatestAudioChunkPublisher::publish_chunk(const AudioChunkView& chunk) {
  return impl_->publish_chunk(chunk);
}

bool ZenohLatestAudioChunkPublisher::valid() const {
  return impl_->valid();
}

std::string ZenohLatestAudioChunkPublisher::key_expr() const {
  return impl_->key_expr();
}

ZenohLatestAudioChunkSubscriber::ZenohLatestAudioChunkSubscriber() : impl_(std::make_unique<Impl>()) {}
ZenohLatestAudioChunkSubscriber::~ZenohLatestAudioChunkSubscriber() {
  close();
}

bool ZenohLatestAudioChunkSubscriber::open(const RuntimeBackendConfig& config, const std::string& key_expr) {
  return impl_->open(config, key_expr);
}

void ZenohLatestAudioChunkSubscriber::close() {
  impl_->close();
}

std::optional<LatestAudioChunk> ZenohLatestAudioChunkSubscriber::poll_latest() {
  return impl_->poll_latest();
}

std::optional<LatestAudioChunk> ZenohLatestAudioChunkSubscriber::wait_latest(std::chrono::milliseconds timeout) {
  return impl_->wait_latest(timeout);
}

bool ZenohLatestAudioChunkSubscriber::valid() const {
  return impl_->valid();
}

std::string ZenohLatestAudioChunkSubscriber::key_expr() const {
  return impl_->key_expr();
}

}  // namespace f8::cppsdk
