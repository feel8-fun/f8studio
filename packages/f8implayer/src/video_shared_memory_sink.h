#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "shm_region.h"

namespace f8::implayer {

class VideoSharedMemorySink {
 public:
  VideoSharedMemorySink() = default;
  ~VideoSharedMemorySink();

  bool initialize(const std::string& region_name, std::size_t capacity_bytes, std::uint32_t slot_count = 2);

  bool ensureConfiguration(unsigned width, unsigned height);
  bool writeFrame(const void* data, unsigned stride_bytes);

  unsigned outputWidth() const { return width_; }
  unsigned outputHeight() const { return height_; }
  unsigned outputPitch() const { return pitch_; }
  std::uint64_t frameId() const { return frame_id_; }

  const std::string& regionName() const { return region_.name(); }
  const std::string& frameEventName() const { return frame_event_name_; }

 private:
  bool configureDimensions(unsigned requested_width, unsigned requested_height);

  ShmRegion region_;
  std::uint32_t slot_count_ = 0;
  std::size_t slot_payload_capacity_ = 0;

  std::string frame_event_name_;

  unsigned width_ = 0;
  unsigned height_ = 0;
  unsigned pitch_ = 0;
  std::uint64_t frame_id_ = 0;
};

}  // namespace f8::implayer

