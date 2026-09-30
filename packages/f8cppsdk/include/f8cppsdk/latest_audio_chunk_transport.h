#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "f8cppsdk/generated/stream_wire.h"
#include "f8cppsdk/runtime_backend.h"
#include "f8cppsdk/runtime_transport.h"

namespace f8::cppsdk {

struct AudioChunkView {
  std::uint32_t sample_rate = 0;
  std::uint32_t channels = 0;
  std::uint32_t format = 0;
  std::uint32_t frames = 0;
  std::uint32_t bytes_per_frame = 0;
  std::uint64_t seq = 0;
  std::uint64_t frame_index = 0;
  std::int64_t ts_ms = 0;
  const void* payload = nullptr;
  std::size_t payload_bytes = 0;
};

struct LatestAudioChunk {
  std::uint32_t sample_rate = 0;
  std::uint32_t channels = 0;
  std::uint32_t format = 0;
  std::uint32_t frames = 0;
  std::uint32_t bytes_per_frame = 0;
  std::uint64_t seq = 0;
  std::uint64_t frame_index = 0;
  std::int64_t ts_ms = 0;
  std::vector<std::byte> payload;
};

bool encode_zenoh_audio_chunk(const AudioChunkView& chunk, RuntimeBytes& out, std::string* error_message = nullptr);
bool decode_zenoh_audio_chunk(const RuntimeBytes& raw, LatestAudioChunk& out, std::string* error_message = nullptr);

class ZenohLatestAudioChunkPublisher final {
 public:
  ZenohLatestAudioChunkPublisher();
  ~ZenohLatestAudioChunkPublisher();
  ZenohLatestAudioChunkPublisher(const ZenohLatestAudioChunkPublisher&) = delete;
  ZenohLatestAudioChunkPublisher& operator=(const ZenohLatestAudioChunkPublisher&) = delete;
  ZenohLatestAudioChunkPublisher(ZenohLatestAudioChunkPublisher&&) = delete;
  ZenohLatestAudioChunkPublisher& operator=(ZenohLatestAudioChunkPublisher&&) = delete;

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr);
  void close();
  bool publish_chunk(const AudioChunkView& chunk);

  bool valid() const;
  std::string key_expr() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class ZenohLatestAudioChunkSubscriber final {
 public:
  ZenohLatestAudioChunkSubscriber();
  ~ZenohLatestAudioChunkSubscriber();
  ZenohLatestAudioChunkSubscriber(const ZenohLatestAudioChunkSubscriber&) = delete;
  ZenohLatestAudioChunkSubscriber& operator=(const ZenohLatestAudioChunkSubscriber&) = delete;
  ZenohLatestAudioChunkSubscriber(ZenohLatestAudioChunkSubscriber&&) = delete;
  ZenohLatestAudioChunkSubscriber& operator=(ZenohLatestAudioChunkSubscriber&&) = delete;

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr);
  void close();
  std::optional<LatestAudioChunk> poll_latest();
  std::optional<LatestAudioChunk> wait_latest(std::chrono::milliseconds timeout);

  bool valid() const;
  std::string key_expr() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace f8::cppsdk
