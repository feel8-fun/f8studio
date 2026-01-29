#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "f8cppsdk/audio_shared_memory_sink.h"
#include "f8cppsdk/shm/naming.h"
#include "f8cppsdk/video_shared_memory_sink.h"

namespace f8::cppsdk::shm {

namespace detail {

inline bool try_mul_size(std::size_t a, std::size_t b, std::size_t& out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > (std::numeric_limits<std::size_t>::max)() / b) return false;
  out = a * b;
  return true;
}

inline bool try_add_size(std::size_t a, std::size_t b, std::size_t& out) {
  if (a > (std::numeric_limits<std::size_t>::max)() - b) return false;
  out = a + b;
  return true;
}

}  // namespace detail

inline std::size_t video_min_bytes(std::uint32_t slot_count = kDefaultVideoShmSlots) {
  if (slot_count == 0) slot_count = 1;
  constexpr std::size_t kMinSlotPayloadBytes = 32 * 32 * 4;
  std::size_t payload = 0;
  if (!detail::try_mul_size(static_cast<std::size_t>(slot_count), kMinSlotPayloadBytes, payload)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t total = 0;
  if (!detail::try_add_size(sizeof(VideoSharedMemoryHeader), payload, total)) return (std::numeric_limits<std::size_t>::max)();
  return total;
}

// Required bytes to guarantee the SHM can hold `slot_count` frames of `max_width`x`max_height` BGRA32.
inline std::size_t video_required_bytes(std::uint32_t max_width, std::uint32_t max_height,
                                        std::uint32_t slot_count = kDefaultVideoShmSlots) {
  if (slot_count == 0) slot_count = 1;
  const std::size_t bpp = 4;

  std::size_t pixels = 0;
  if (!detail::try_mul_size(static_cast<std::size_t>(max_width), static_cast<std::size_t>(max_height), pixels)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t per_frame = 0;
  if (!detail::try_mul_size(pixels, bpp, per_frame)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t payload = 0;
  if (!detail::try_mul_size(static_cast<std::size_t>(slot_count), per_frame, payload)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t total = 0;
  if (!detail::try_add_size(sizeof(VideoSharedMemoryHeader), payload, total)) return (std::numeric_limits<std::size_t>::max)();

  // Respect minimum slot payload.
  return std::max(total, video_min_bytes(slot_count));
}

inline std::size_t video_recommended_bytes(std::uint32_t max_width, std::uint32_t max_height,
                                           std::uint32_t slot_count = kDefaultVideoShmSlots) {
  return std::max(video_required_bytes(max_width, max_height, slot_count), kDefaultVideoShmBytes);
}

// Required bytes for AudioSHM ring buffer.
inline std::size_t audio_required_bytes(std::uint32_t sample_rate, std::uint16_t channels, std::uint32_t frames_per_chunk,
                                        std::uint32_t chunk_count,
                                        AudioSharedMemorySink::SampleFormat format = AudioSharedMemorySink::SampleFormat::kF32LE) {
  (void)sample_rate;
  if (channels == 0 || frames_per_chunk == 0 || chunk_count == 0) return 0;

  std::size_t bytes_per_sample = 0;
  switch (format) {
    case AudioSharedMemorySink::SampleFormat::kF32LE:
      bytes_per_sample = 4;
      break;
    case AudioSharedMemorySink::SampleFormat::kS16LE:
      bytes_per_sample = 2;
      break;
    default:
      return 0;
  }

  std::size_t bytes_per_frame = 0;
  if (!detail::try_mul_size(bytes_per_sample, static_cast<std::size_t>(channels), bytes_per_frame)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t payload_per_chunk = 0;
  if (!detail::try_mul_size(bytes_per_frame, static_cast<std::size_t>(frames_per_chunk), payload_per_chunk)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t chunk_stride = 0;
  if (!detail::try_add_size(sizeof(AudioSharedMemoryChunkHeader), payload_per_chunk, chunk_stride)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t chunks_total = 0;
  if (!detail::try_mul_size(chunk_stride, static_cast<std::size_t>(chunk_count), chunks_total)) return (std::numeric_limits<std::size_t>::max)();
  std::size_t total = 0;
  if (!detail::try_add_size(sizeof(AudioSharedMemoryHeader), chunks_total, total)) return (std::numeric_limits<std::size_t>::max)();
  return total;
}

inline std::size_t audio_recommended_bytes(std::uint32_t sample_rate, std::uint16_t channels, std::uint32_t frames_per_chunk,
                                           std::uint32_t chunk_count,
                                           AudioSharedMemorySink::SampleFormat format = AudioSharedMemorySink::SampleFormat::kF32LE) {
  return std::max(audio_required_bytes(sample_rate, channels, frames_per_chunk, chunk_count, format), kDefaultAudioShmBytes);
}

}  // namespace f8::cppsdk::shm

