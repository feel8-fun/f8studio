#include "audio_shared_memory_sink.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include <spdlog/spdlog.h>

#if defined(_WIN32)
#include <Windows.h>
#endif

namespace f8::audiocap {

namespace {

constexpr std::uint32_t kShmMagic = 0xF8A11A02u;
constexpr std::uint32_t kShmVersion = 1u;

struct AudioShmHeader {
  std::uint32_t magic = kShmMagic;
  std::uint32_t version = kShmVersion;
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

struct AudioChunkHeader {
  std::uint64_t seq = 0;
  std::int64_t ts_ms = 0;
  std::uint32_t frames = 0;
  std::uint32_t reserved = 0;
};

inline std::size_t header_size() { return sizeof(AudioShmHeader); }
inline std::size_t chunk_header_size() { return sizeof(AudioChunkHeader); }

inline std::uint32_t bytes_per_sample(AudioSharedMemorySink::SampleFormat fmt) {
  switch (fmt) {
    case AudioSharedMemorySink::SampleFormat::kF32LE:
      return 4;
    case AudioSharedMemorySink::SampleFormat::kS16LE:
      return 2;
    default:
      return 0;
  }
}

}  // namespace

struct AudioSharedMemorySink::Header : AudioShmHeader {};
struct AudioSharedMemorySink::ChunkHeader : AudioChunkHeader {};

AudioSharedMemorySink::~AudioSharedMemorySink() {
#if defined(_WIN32)
  if (frame_event_) {
    CloseHandle(static_cast<HANDLE>(frame_event_));
    frame_event_ = nullptr;
  }
#endif
}

bool AudioSharedMemorySink::initialize(const std::string& region_name, std::size_t capacity_bytes, std::uint32_t sample_rate,
                                      std::uint16_t channels, SampleFormat format, std::uint32_t frames_per_chunk,
                                      std::uint32_t chunk_count) {
  if (region_name.empty()) return false;
  if (sample_rate == 0 || channels == 0 || frames_per_chunk == 0 || chunk_count == 0) return false;

  const std::uint32_t bps = bytes_per_sample(format);
  if (bps == 0) return false;

  const std::uint32_t bytes_per_frame = bps * static_cast<std::uint32_t>(channels);
  const std::size_t payload_bytes_per_chunk = static_cast<std::size_t>(bytes_per_frame) * frames_per_chunk;
  const std::size_t chunk_stride = chunk_header_size() + payload_bytes_per_chunk;
  const std::size_t required = header_size() + chunk_stride * chunk_count;
  if (capacity_bytes < required) {
    spdlog::error("audio shm capacity too small required={} actual={}", required, capacity_bytes);
    return false;
  }

  if (!region_.open_or_create(region_name, capacity_bytes)) return false;

  sample_rate_ = sample_rate;
  channels_ = channels;
  format_ = format;
  frames_per_chunk_ = frames_per_chunk;
  chunk_count_ = chunk_count;
  bytes_per_frame_ = bytes_per_frame;
  chunk_stride_bytes_ = chunk_stride;
  payload_offset_bytes_ = chunk_header_size();

#if defined(_WIN32)
  frame_event_name_ = region_name + "_evt";
  if (frame_event_) {
    CloseHandle(static_cast<HANDLE>(frame_event_));
    frame_event_ = nullptr;
  }
  {
    const std::wstring wname(frame_event_name_.begin(), frame_event_name_.end());
    HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, wname.c_str());
    if (!ev) {
      spdlog::warn("CreateEventW failed name={} err={}", frame_event_name_, GetLastError());
    } else {
      frame_event_ = ev;
    }
  }
#endif

  auto* hdr = static_cast<AudioShmHeader*>(region_.data());
  if (!hdr) return false;
  std::memset(hdr, 0, sizeof(*hdr));
  hdr->magic = kShmMagic;
  hdr->version = kShmVersion;
  hdr->sample_rate = sample_rate_;
  hdr->channels = channels_;
  hdr->format = static_cast<std::uint16_t>(format_);
  hdr->frames_per_chunk = frames_per_chunk_;
  hdr->chunk_count = chunk_count_;
  hdr->bytes_per_frame = bytes_per_frame_;
  hdr->payload_bytes_per_chunk = static_cast<std::uint32_t>(payload_bytes_per_chunk);
  hdr->write_seq = 0;
  hdr->write_frame_index = 0;
  hdr->ts_ms = 0;

  write_seq_.store(0, std::memory_order_relaxed);
  write_frame_index_.store(0, std::memory_order_relaxed);

  std::byte* base = static_cast<std::byte*>(region_.data());
  std::memset(base + header_size(), 0, chunk_stride_bytes_ * chunk_count_);
  return true;
}

bool AudioSharedMemorySink::write_interleaved_f32(const float* samples, std::uint32_t frames, std::int64_t ts_ms) {
  if (!samples) return false;
  if (!region_.data()) return false;
  if (format_ != SampleFormat::kF32LE) return false;
  if (frames == 0) return true;

  const std::uint32_t max_frames = frames_per_chunk_;
  const std::uint32_t write_frames = std::min(frames, max_frames);
  const std::size_t bytes = static_cast<std::size_t>(write_frames) * bytes_per_frame_;

  auto* hdr = static_cast<AudioShmHeader*>(region_.data());
  if (!hdr || hdr->magic != kShmMagic || hdr->version != kShmVersion) return false;

  const std::uint64_t seq = write_seq_.load(std::memory_order_relaxed) + 1;
  const std::uint32_t idx = static_cast<std::uint32_t>(seq % chunk_count_);

  std::byte* base = static_cast<std::byte*>(region_.data());
  std::byte* chunk = base + header_size() + static_cast<std::size_t>(idx) * chunk_stride_bytes_;
  auto* chdr = reinterpret_cast<AudioChunkHeader*>(chunk);
  std::byte* payload = chunk + payload_offset_bytes_;

  std::memcpy(payload, samples, bytes);
  std::atomic_thread_fence(std::memory_order_release);

  chdr->ts_ms = ts_ms;
  chdr->frames = write_frames;
  std::atomic_thread_fence(std::memory_order_release);
  chdr->seq = seq;

  const std::uint64_t frame_index = write_frame_index_.fetch_add(write_frames, std::memory_order_relaxed) + write_frames;
  write_seq_.store(seq, std::memory_order_relaxed);

  hdr->write_frame_index = frame_index;
  hdr->ts_ms = ts_ms;
  hdr->write_seq = seq;

#if defined(_WIN32)
  if (frame_event_) {
    SetEvent(static_cast<HANDLE>(frame_event_));
    ResetEvent(static_cast<HANDLE>(frame_event_));
  }
#endif

  return true;
}

}  // namespace f8::audiocap

