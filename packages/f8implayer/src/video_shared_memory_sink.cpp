#include "video_shared_memory_sink.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>

#include <spdlog/spdlog.h>

namespace f8::implayer {

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

VideoSharedMemorySink::~VideoSharedMemorySink() = default;

bool VideoSharedMemorySink::initialize(const std::string& region_name, std::size_t capacity_bytes,
                                      std::uint32_t slot_count) {
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

  auto* hdr = static_cast<ShmHeader*>(region_.data());
  if (!hdr) return false;
  hdr->magic = kShmMagic;
  hdr->version = 1;
  hdr->slot_count = slot_count_;

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
  hdr->active_slot = slot;
  hdr->frame_id = ++frame_id_;
  hdr->ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
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

}  // namespace f8::implayer
