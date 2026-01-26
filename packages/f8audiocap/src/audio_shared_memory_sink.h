#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "shm_region.h"

namespace f8::audiocap {

class AudioSharedMemorySink {
 public:
  enum class SampleFormat : std::uint16_t {
    kF32LE = 1,
    kS16LE = 2,
  };

  ~AudioSharedMemorySink();

  bool initialize(const std::string& region_name, std::size_t capacity_bytes, std::uint32_t sample_rate,
                  std::uint16_t channels, SampleFormat format, std::uint32_t frames_per_chunk,
                  std::uint32_t chunk_count);

  bool write_interleaved_f32(const float* samples, std::uint32_t frames, std::int64_t ts_ms);

  const std::string& shm_name() const { return region_.name(); }
  std::uint64_t write_seq() const { return write_seq_.load(std::memory_order_relaxed); }

 private:
  struct Header;
  struct ChunkHeader;

  ShmRegion region_;
  std::size_t chunk_stride_bytes_ = 0;
  std::size_t payload_offset_bytes_ = 0;

  std::uint32_t sample_rate_ = 0;
  std::uint16_t channels_ = 0;
  SampleFormat format_ = SampleFormat::kF32LE;
  std::uint32_t frames_per_chunk_ = 0;
  std::uint32_t chunk_count_ = 0;
  std::uint32_t bytes_per_frame_ = 0;

  std::atomic<std::uint64_t> write_seq_{0};
  std::atomic<std::uint64_t> write_frame_index_{0};

#if defined(_WIN32)
  void* frame_event_ = nullptr;
  std::string frame_event_name_;
#endif
};

}  // namespace f8::audiocap

