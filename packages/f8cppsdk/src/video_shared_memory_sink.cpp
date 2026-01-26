#include "f8cppsdk/video_shared_memory_sink.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include <spdlog/spdlog.h>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#endif

namespace f8::cppsdk {

namespace {

constexpr std::uint32_t kShmMagic = 0xF8A11A01u;
constexpr std::size_t kMinSlotPayloadBytes = 32 * 32 * 4;

struct ShmHeader {
  std::uint32_t magic = kShmMagic;
  std::uint32_t version = 1;
  std::uint32_t slot_count = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t pitch = 0;
  std::uint32_t format = 1;  // 1=BGRA32
  std::uint64_t frame_id = 0;
  std::int64_t ts_ms = 0;
  std::uint32_t active_slot = 0;
  std::uint32_t payload_capacity = 0;
};

inline std::size_t header_size() { return sizeof(ShmHeader); }

}  // namespace

VideoSharedMemorySink::~VideoSharedMemorySink() {
#if defined(_WIN32)
  if (frame_event_) {
    CloseHandle(static_cast<HANDLE>(frame_event_));
    frame_event_ = nullptr;
  }
#endif
}

bool VideoSharedMemorySink::initialize(const std::string& region_name, std::size_t capacity_bytes, std::uint32_t slot_count) {
  if (slot_count == 0) slot_count = 1;
  if (capacity_bytes < header_size() + kMinSlotPayloadBytes) {
    spdlog::error("video shm capacity too small");
    return false;
  }
  if (!region_.open_or_create(region_name, capacity_bytes)) {
    return false;
  }
  slot_count_ = slot_count;
  frame_event_name_ = region_name + "_evt";
#if defined(_WIN32)
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

  auto* hdr = static_cast<ShmHeader*>(region_.data());
  if (!hdr) return false;
  hdr->magic = kShmMagic;
  hdr->version = 1;
  hdr->slot_count = slot_count_;
  hdr->format = 1;
  hdr->frame_id = 0;
  hdr->ts_ms = 0;
  hdr->active_slot = 0;
  hdr->width = 0;
  hdr->height = 0;
  hdr->pitch = 0;

  const std::size_t usable = capacity_bytes - header_size();
  slot_payload_capacity_ = usable / slot_count_;
  if (slot_payload_capacity_ < kMinSlotPayloadBytes) {
    spdlog::error("video shm payload capacity too small per slot");
    region_.close();
    return false;
  }
  hdr->payload_capacity = static_cast<std::uint32_t>(slot_payload_capacity_);
  return true;
}

bool VideoSharedMemorySink::ensureConfiguration(unsigned width, unsigned height) { return configureDimensions(width, height); }

bool VideoSharedMemorySink::writeFrame(const void* data, unsigned stride_bytes) {
  if (!data || width_ == 0 || height_ == 0 || pitch_ == 0) return false;
  if (!region_.data()) return false;
  auto* hdr = static_cast<ShmHeader*>(region_.data());
  if (!hdr || hdr->magic != kShmMagic) return false;

  const std::size_t row_bytes = static_cast<std::size_t>(pitch_);
  const std::size_t payload_bytes = row_bytes * height_;
  if (payload_bytes == 0 || payload_bytes > slot_payload_capacity_) return false;
  if (static_cast<std::size_t>(stride_bytes) < row_bytes) return false;

  const std::uint32_t slot = (hdr->active_slot + 1) % std::max<std::uint32_t>(1, slot_count_);
  std::byte* base = static_cast<std::byte*>(region_.data());
  std::byte* dst = base + header_size() + static_cast<std::size_t>(slot) * slot_payload_capacity_;

  const auto* src = static_cast<const std::byte*>(data);
  const std::size_t src_stride = static_cast<std::size_t>(stride_bytes);
  for (unsigned y = 0; y < height_; ++y) {
    std::memcpy(dst + static_cast<std::size_t>(y) * row_bytes, src + static_cast<std::size_t>(y) * src_stride, row_bytes);
  }

  hdr->width = width_;
  hdr->height = height_;
  hdr->pitch = pitch_;
  hdr->format = 1;
  hdr->active_slot = slot;
  hdr->frame_id = ++frame_id_;
  hdr->ts_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

#if defined(_WIN32)
  if (frame_event_) {
    SetEvent(static_cast<HANDLE>(frame_event_));
    ResetEvent(static_cast<HANDLE>(frame_event_));
  }
#endif

  return true;
}

bool VideoSharedMemorySink::configureDimensions(unsigned requested_width, unsigned requested_height) {
  if (requested_width == 0 || requested_height == 0) return false;

  auto align32 = [](unsigned v) { return v < 32u ? 32u : (v / 32u) * 32u; };
  unsigned out_w = align32(requested_width);
  unsigned out_h = align32(requested_height);

  const unsigned bpp = 4;
  auto required_bytes = [&]() { return static_cast<std::size_t>(out_w) * out_h * bpp; };

  while (required_bytes() > slot_payload_capacity_) {
    const double ratio = std::sqrt(static_cast<double>(slot_payload_capacity_) / static_cast<double>(required_bytes()));
    unsigned new_w = align32(static_cast<unsigned>(static_cast<double>(out_w) * ratio));
    unsigned new_h = align32(static_cast<unsigned>(static_cast<double>(out_h) * ratio));
    new_w = std::max(new_w, 32u);
    new_h = std::max(new_h, 32u);
    if (new_w == out_w && new_h == out_h) return false;
    out_w = new_w;
    out_h = new_h;
  }

  if (out_w == width_ && out_h == height_) return true;
  width_ = out_w;
  height_ = out_h;
  pitch_ = out_w * bpp;
  return true;
}

bool VideoSharedMemoryReader::open(const std::string& region_name, std::size_t bytes) {
  return region_.open_existing_readonly(region_name, bytes);
}

bool VideoSharedMemoryReader::readHeader(VideoSharedMemoryHeader& out) const {
  if (!region_.data() || region_.size() < sizeof(ShmHeader)) return false;
  const auto* hdr = static_cast<const ShmHeader*>(region_.data());
  if (hdr->magic != kShmMagic) return false;
  out.magic = hdr->magic;
  out.version = hdr->version;
  out.slot_count = hdr->slot_count;
  out.width = hdr->width;
  out.height = hdr->height;
  out.pitch = hdr->pitch;
  out.format = hdr->format;
  out.frame_id = hdr->frame_id;
  out.ts_ms = hdr->ts_ms;
  out.active_slot = hdr->active_slot;
  out.payload_capacity = hdr->payload_capacity;
  return true;
}

bool VideoSharedMemoryReader::copyLatestFrame(std::vector<std::byte>& out_bgra, VideoSharedMemoryHeader& out_header) const {
  if (!region_.data() || region_.size() < sizeof(ShmHeader)) return false;

  VideoSharedMemoryHeader h0{};
  VideoSharedMemoryHeader h1{};
  if (!readHeader(h0)) return false;
  if (h0.width == 0 || h0.height == 0 || h0.pitch == 0 || h0.payload_capacity == 0) return false;
  if (static_cast<std::size_t>(h0.pitch) * h0.height > h0.payload_capacity) return false;

  // Simple stability check: read twice and require frame_id/slot match.
  if (!readHeader(h1)) return false;
  if (h1.frame_id != h0.frame_id || h1.active_slot != h0.active_slot) return false;

  const std::size_t header_bytes = sizeof(ShmHeader);
  const std::size_t slot_off = header_bytes + static_cast<std::size_t>(h0.active_slot) * h0.payload_capacity;
  const std::size_t bytes = static_cast<std::size_t>(h0.pitch) * h0.height;
  if (slot_off + bytes > region_.size()) return false;

  out_bgra.resize(bytes);
  const std::byte* base = static_cast<const std::byte*>(region_.data());
  std::memcpy(out_bgra.data(), base + slot_off, bytes);
  out_header = h0;
  return true;
}

}  // namespace f8::cppsdk
