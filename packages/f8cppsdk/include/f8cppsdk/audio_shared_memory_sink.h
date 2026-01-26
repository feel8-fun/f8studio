#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "f8cppsdk/shm_region.h"

namespace f8::cppsdk {

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
  const std::string& frameEventName() const { return frame_event_name_; }

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

  std::string frame_event_name_;
#if defined(_WIN32)
  void* frame_event_ = nullptr;
#endif
};

struct AudioSharedMemoryHeader {
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t sample_rate = 0;
  std::uint16_t channels = 0;
  std::uint16_t format = 0;
  std::uint32_t frames_per_chunk = 0;
  std::uint32_t chunk_count = 0;
  std::uint32_t bytes_per_frame = 0;
  std::uint32_t payload_bytes_per_chunk = 0;
  std::uint64_t write_seq = 0;
  std::uint64_t write_frame_index = 0;
  std::int64_t ts_ms = 0;
};

struct AudioSharedMemoryChunkHeader {
  std::uint64_t seq = 0;
  std::int64_t ts_ms = 0;
  std::uint32_t frames = 0;
};

class AudioSharedMemoryReader {
 public:
  AudioSharedMemoryReader() = default;
  ~AudioSharedMemoryReader() = default;

  bool open(const std::string& region_name, std::size_t bytes);
  void close() { region_.close(); }

  bool readHeader(AudioSharedMemoryHeader& out) const;
  bool readChunkF32(std::uint64_t seq, std::vector<float>& out_interleaved, AudioSharedMemoryChunkHeader& out_chunk,
                    AudioSharedMemoryHeader& out_header) const;

 private:
  ShmRegion region_;
};

}  // namespace f8::cppsdk

