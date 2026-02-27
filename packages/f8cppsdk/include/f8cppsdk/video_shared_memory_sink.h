#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "f8cppsdk/shm_region.h"

namespace f8::cppsdk {

constexpr std::uint32_t kVideoFormatBgra32 = 1;
constexpr std::uint32_t kVideoFormatFlow2F16 = 2;

class VideoSharedMemorySink {
 public:
  VideoSharedMemorySink() = default;
  ~VideoSharedMemorySink();

  bool initialize(const std::string& region_name, std::size_t capacity_bytes, std::uint32_t slot_count = 2);
  // POSIX only: controls whether the creator unlinks the SHM name on close().
  // Default is false; enabling this can break other processes that want to attach later.
  void set_unlink_on_close(bool enabled) { region_.set_unlink_on_close(enabled); }
  bool ensureConfiguration(unsigned width, unsigned height);
  bool ensureConfigurationForFormat(unsigned width, unsigned height, std::uint32_t format, unsigned bytes_per_pixel);
  bool writeFrame(const void* data, unsigned stride_bytes);
  bool writeFrameWithFormat(const void* data, unsigned stride_bytes, std::uint32_t format);

  unsigned outputWidth() const { return width_; }
  unsigned outputHeight() const { return height_; }
  unsigned outputPitch() const { return pitch_; }
  std::uint32_t outputFormat() const { return format_; }
  unsigned outputBytesPerPixel() const { return bytes_per_pixel_; }
  std::uint64_t frameId() const { return frame_id_; }

  const std::string& regionName() const { return region_.name(); }
  const std::string& frameEventName() const { return frame_event_name_; }

 private:
  bool configureDimensions(unsigned requested_width, unsigned requested_height, unsigned bytes_per_pixel);

  ShmRegion region_;
  std::uint32_t slot_count_ = 0;
  std::size_t slot_payload_capacity_ = 0;

  std::string frame_event_name_;
#if defined(_WIN32)
  void* frame_event_ = nullptr;
#endif

  unsigned width_ = 0;
  unsigned height_ = 0;
  unsigned pitch_ = 0;
  unsigned bytes_per_pixel_ = 4;
  std::uint32_t format_ = kVideoFormatBgra32;
  std::uint64_t frame_id_ = 0;
};

struct VideoSharedMemoryHeader {
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t slot_count = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t pitch = 0;
  std::uint32_t format = 0;
  std::uint64_t frame_id = 0;
  std::int64_t ts_ms = 0;
  std::uint32_t active_slot = 0;
  std::uint32_t payload_capacity = 0;
  std::uint32_t notify_seq = 0;
};

class VideoSharedMemoryReader {
 public:
  VideoSharedMemoryReader() = default;
  ~VideoSharedMemoryReader();

  bool open(const std::string& region_name, std::size_t bytes);
  void close();

  bool readHeader(VideoSharedMemoryHeader& out) const;
  bool waitNewFrame(std::uint32_t last_notify_seq, std::uint32_t timeout_ms, std::uint32_t* observed_notify_seq) const;
  bool copyLatestPayload(std::vector<std::byte>& out_payload, VideoSharedMemoryHeader& out_header) const;
  bool copyLatestFrame(std::vector<std::byte>& out_bgra, VideoSharedMemoryHeader& out_header) const;

 private:
  ShmRegion region_;
#if defined(_WIN32)
  void* frame_event_ = nullptr;
#endif
};

}  // namespace f8::cppsdk
