#include "f8cppsdk/video_shared_memory_sink.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <string>

#include <spdlog/spdlog.h>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace f8::cppsdk {

namespace {

constexpr std::uint32_t kShmMagic = 0xF8A11A01u;
constexpr std::size_t kMinSlotPayloadBytes = 32 * 32 * 4;

bool shm_unlink_on_close_enabled() {
  const char* v = std::getenv("F8_SHM_UNLINK_ON_CLOSE");
  if (v == nullptr) return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

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
  std::uint32_t notify_seq = 0;
  std::uint32_t reserved = 0;
};
static_assert(sizeof(ShmHeader) == 64, "Video SHM header size mismatch");

inline std::size_t header_size() { return sizeof(ShmHeader); }

inline std::uint32_t notify_seq_add_release(std::uint32_t* addr) {
#if defined(_WIN32)
  return static_cast<std::uint32_t>(InterlockedIncrement(reinterpret_cast<volatile long*>(addr)));
#elif defined(__linux__)
  return __atomic_add_fetch(addr, 1u, __ATOMIC_RELEASE);
#else
  return __atomic_add_fetch(addr, 1u, __ATOMIC_RELEASE);
#endif
}

inline std::uint32_t notify_seq_load_acquire(const std::uint32_t* addr) {
#if defined(_WIN32)
  // Reader maps SHM as read-only; do not use interlocked RMW ops here.
  // Use a volatile load followed by an acquire fence to preserve ordering.
  const auto* src = reinterpret_cast<volatile const std::uint32_t*>(addr);
  const std::uint32_t v = *src;
  std::atomic_thread_fence(std::memory_order_acquire);
  return v;
#elif defined(__linux__)
  return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
#else
  return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
#endif
}

#if defined(__linux__)
int futex_wait_u32(const std::uint32_t* addr, std::uint32_t expected, std::uint32_t timeout_ms) {
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(timeout_ms / 1000);
  ts.tv_nsec = static_cast<long>((timeout_ms % 1000) * 1000000u);
  return static_cast<int>(syscall(SYS_futex, reinterpret_cast<const int*>(addr), FUTEX_WAIT, static_cast<int>(expected),
                                  &ts, nullptr, 0));
}

int futex_wake_all_u32(std::uint32_t* addr) {
  return static_cast<int>(
      syscall(SYS_futex, reinterpret_cast<int*>(addr), FUTEX_WAKE, static_cast<int>(INT_MAX), nullptr, nullptr, 0));
}
#endif

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
#if !defined(_WIN32)
  // POSIX SHM names persist until shm_unlink().
  // Default is to NOT unlink on close so other processes can still attach/recover.
  // Opt-in via env var for dev/test cleanup.
  region_.set_unlink_on_close(shm_unlink_on_close_enabled());
#endif
  slot_count_ = slot_count;
  frame_event_name_ = region_name + "_evt";
#if defined(_WIN32)
  if (frame_event_) {
    CloseHandle(static_cast<HANDLE>(frame_event_));
    frame_event_ = nullptr;
  }
  {
    const std::wstring wname(frame_event_name_.begin(), frame_event_name_.end());
    // Auto-reset event so slow consumers don't miss signals.
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, wname.c_str());
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
  hdr->format = kVideoFormatBgra32;
  hdr->frame_id = 0;
  hdr->ts_ms = 0;
  hdr->active_slot = 0;
  hdr->width = 0;
  hdr->height = 0;
  hdr->pitch = 0;
  hdr->notify_seq = 0;
  hdr->reserved = 0;

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

bool VideoSharedMemorySink::writeFrame(const void* data, unsigned stride_bytes) {
  return writeFrameWithFormat(data, stride_bytes, kVideoFormatBgra32);
}

bool VideoSharedMemorySink::ensureConfigurationForFormat(unsigned width, unsigned height, std::uint32_t format,
                                                         unsigned bytes_per_pixel) {
  if (bytes_per_pixel == 0) return false;
  if (format == 0) return false;
  if (!configureDimensions(width, height, bytes_per_pixel)) return false;
  format_ = format;
  return true;
}

bool VideoSharedMemorySink::writeFrameWithFormat(const void* data, unsigned stride_bytes, std::uint32_t format) {
  if (!data || width_ == 0 || height_ == 0 || pitch_ == 0) return false;
  if (!region_.data()) return false;
  if (format_ != format) return false;
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
  hdr->format = format_;
  hdr->active_slot = slot;
  hdr->frame_id = ++frame_id_;
  hdr->ts_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  (void)notify_seq_add_release(&hdr->notify_seq);

#if defined(_WIN32)
  if (frame_event_) {
    SetEvent(static_cast<HANDLE>(frame_event_));
  }
#elif defined(__linux__)
  (void)futex_wake_all_u32(&hdr->notify_seq);
#endif

  return true;
}

bool VideoSharedMemorySink::configureDimensions(unsigned requested_width, unsigned requested_height, unsigned bytes_per_pixel) {
  if (requested_width == 0 || requested_height == 0) return false;
  if (bytes_per_pixel == 0) return false;

  // Keep requested dimensions to avoid unnecessary resampling artifacts.
  unsigned out_w = requested_width;
  unsigned out_h = requested_height;
  auto required_bytes = [&]() { return static_cast<std::size_t>(out_w) * out_h * bytes_per_pixel; };

  while (required_bytes() > slot_payload_capacity_) {
    const double ratio = std::sqrt(static_cast<double>(slot_payload_capacity_) / static_cast<double>(required_bytes()));
    unsigned new_w = static_cast<unsigned>(static_cast<double>(out_w) * ratio);
    unsigned new_h = static_cast<unsigned>(static_cast<double>(out_h) * ratio);
    new_w = std::max(new_w, 1u);
    new_h = std::max(new_h, 1u);
    if (new_w == out_w && new_h == out_h) return false;
    out_w = new_w;
    out_h = new_h;
  }

  if (out_w == width_ && out_h == height_) return true;
  width_ = out_w;
  height_ = out_h;
  bytes_per_pixel_ = bytes_per_pixel;
  pitch_ = out_w * bytes_per_pixel_;
  return true;
}

bool VideoSharedMemorySink::ensureConfiguration(unsigned width, unsigned height) {
  return ensureConfigurationForFormat(width, height, kVideoFormatBgra32, 4);
}

bool VideoSharedMemoryReader::open(const std::string& region_name, std::size_t bytes) {
  close();
  if (!region_.open_existing_readonly(region_name, bytes)) {
    return false;
  }
#if defined(_WIN32)
  const std::string ev_name = region_name + "_evt";
  const std::wstring wname(ev_name.begin(), ev_name.end());
  HANDLE h = OpenEventW(SYNCHRONIZE, FALSE, wname.c_str());
  frame_event_ = h;
#endif
  return true;
}

VideoSharedMemoryReader::~VideoSharedMemoryReader() { close(); }

void VideoSharedMemoryReader::close() {
#if defined(_WIN32)
  if (frame_event_) {
    CloseHandle(static_cast<HANDLE>(frame_event_));
    frame_event_ = nullptr;
  }
#endif
  region_.close();
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
  out.notify_seq = notify_seq_load_acquire(&hdr->notify_seq);
  return true;
}

bool VideoSharedMemoryReader::waitNewFrame(std::uint32_t last_notify_seq, std::uint32_t timeout_ms,
                                           std::uint32_t* observed_notify_seq) const {
  if (!region_.data() || region_.size() < sizeof(ShmHeader)) return false;
  const auto* hdr = static_cast<const ShmHeader*>(region_.data());
  if (hdr->magic != kShmMagic) return false;

  const auto load_notify = [&]() { return notify_seq_load_acquire(&hdr->notify_seq); };
  std::uint32_t now_notify_seq = load_notify();
  if (observed_notify_seq) {
    *observed_notify_seq = now_notify_seq;
  }
  if (now_notify_seq != last_notify_seq) {
    return true;
  }

#if defined(_WIN32)
  if (!frame_event_) return false;
  const DWORD rc = WaitForSingleObject(static_cast<HANDLE>(frame_event_), static_cast<DWORD>(timeout_ms));
  if (rc != WAIT_OBJECT_0 && rc != WAIT_TIMEOUT) {
    return false;
  }
#elif defined(__linux__)
  const int rc = futex_wait_u32(&hdr->notify_seq, last_notify_seq, timeout_ms);
  if (rc != 0) {
    const int err = errno;
    if (err != ETIMEDOUT && err != EAGAIN && err != EINTR) {
      return false;
    }
  }
#else
  (void)timeout_ms;
  return false;
#endif

  now_notify_seq = load_notify();
  if (observed_notify_seq) {
    *observed_notify_seq = now_notify_seq;
  }
  return now_notify_seq != last_notify_seq;
}

bool VideoSharedMemoryReader::copyLatestPayload(std::vector<std::byte>& out_payload, VideoSharedMemoryHeader& out_header) const {
  if (!region_.data() || region_.size() < sizeof(ShmHeader)) return false;

  VideoSharedMemoryHeader h0{};
  VideoSharedMemoryHeader h1{};
  if (!readHeader(h0)) return false;
  if (h0.width == 0 || h0.height == 0 || h0.pitch == 0 || h0.payload_capacity == 0) return false;
  if (static_cast<std::size_t>(h0.pitch) * h0.height > h0.payload_capacity) return false;

  // Simple stability check: read twice and require frame_id/slot match.
  if (!readHeader(h1)) return false;
  if (h1.frame_id != h0.frame_id || h1.active_slot != h0.active_slot || h1.notify_seq != h0.notify_seq) return false;

  const std::size_t header_bytes = sizeof(ShmHeader);
  const std::size_t slot_off = header_bytes + static_cast<std::size_t>(h0.active_slot) * h0.payload_capacity;
  const std::size_t bytes = static_cast<std::size_t>(h0.pitch) * h0.height;
  if (slot_off + bytes > region_.size()) return false;

  out_payload.resize(bytes);
  const std::byte* base = static_cast<const std::byte*>(region_.data());
  std::memcpy(out_payload.data(), base + slot_off, bytes);
  out_header = h0;
  return true;
}

bool VideoSharedMemoryReader::copyLatestFrame(std::vector<std::byte>& out_bgra, VideoSharedMemoryHeader& out_header) const {
  if (!copyLatestPayload(out_bgra, out_header)) return false;
  if (out_header.format != kVideoFormatBgra32) return false;
  return true;
}

}  // namespace f8::cppsdk
