#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "f8cppsdk/runtime_backend.h"
#include "f8cppsdk/runtime_transport.h"

namespace f8::cppsdk {

constexpr std::uint32_t kVideoFormatBgra32 = 1;
constexpr std::uint32_t kVideoFormatFlow2F16 = 2;
constexpr std::uint32_t kVideoFormatScalar1F32 = 3;

inline constexpr std::uint32_t kZenohVideoFrameMagic = 0xF85A1001u;
inline constexpr std::uint32_t kZenohVideoFrameSchemaVersion = 2u;
inline constexpr std::uint32_t kZenohVideoFrameHeaderBytes = 64u;

struct VideoFrameView {
  unsigned width = 0;
  unsigned height = 0;
  unsigned pitch = 0;
  std::uint32_t format = 0;
  std::uint64_t frame_id = 0;
  std::int64_t ts_ms = 0;
  const std::byte* payload = nullptr;
  std::size_t payload_bytes = 0;
  std::uint64_t stream_epoch_high = 0;
  std::uint64_t stream_epoch_low = 0;
};

struct LatestVideoFrame {
  unsigned width = 0;
  unsigned height = 0;
  unsigned pitch = 0;
  std::uint32_t format = 0;
  std::uint64_t frame_id = 0;
  std::int64_t ts_ms = 0;
  std::vector<std::byte> payload;
  std::uint64_t stream_epoch_high = 0;
  std::uint64_t stream_epoch_low = 0;
};

bool encode_zenoh_video_frame(const VideoFrameView& frame, RuntimeBytes& out, std::string* error_message = nullptr);
bool decode_zenoh_video_frame(const RuntimeBytes& raw, LatestVideoFrame& out, std::string* error_message = nullptr);

class ZenohLatestVideoFramePublisher final {
 public:
  ZenohLatestVideoFramePublisher();
  ~ZenohLatestVideoFramePublisher();
  ZenohLatestVideoFramePublisher(const ZenohLatestVideoFramePublisher&) = delete;
  ZenohLatestVideoFramePublisher& operator=(const ZenohLatestVideoFramePublisher&) = delete;
  ZenohLatestVideoFramePublisher(ZenohLatestVideoFramePublisher&&) = delete;
  ZenohLatestVideoFramePublisher& operator=(ZenohLatestVideoFramePublisher&&) = delete;

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr);
  void close();
  bool publish_frame(const VideoFrameView& frame);

  bool valid() const;
  std::string key_expr() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class ZenohLatestVideoFrameSubscriber final {
 public:
  ZenohLatestVideoFrameSubscriber();
  ~ZenohLatestVideoFrameSubscriber();
  ZenohLatestVideoFrameSubscriber(const ZenohLatestVideoFrameSubscriber&) = delete;
  ZenohLatestVideoFrameSubscriber& operator=(const ZenohLatestVideoFrameSubscriber&) = delete;
  ZenohLatestVideoFrameSubscriber(ZenohLatestVideoFrameSubscriber&&) = delete;
  ZenohLatestVideoFrameSubscriber& operator=(ZenohLatestVideoFrameSubscriber&&) = delete;

  bool open(const RuntimeBackendConfig& config, const std::string& key_expr);
  void close();
  std::optional<LatestVideoFrame> poll_latest();
  std::optional<LatestVideoFrame> wait_latest(std::chrono::milliseconds timeout);

  bool valid() const;
  std::string key_expr() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace f8::cppsdk
