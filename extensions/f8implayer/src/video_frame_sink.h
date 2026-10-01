#pragma once

#include <cstdint>

namespace f8::implayer {

class VideoFrameSink {
 public:
  virtual ~VideoFrameSink() = default;

  virtual bool ensureConfiguration(unsigned width, unsigned height) = 0;
  virtual bool writeFrame(const void* data, unsigned stride_bytes) = 0;

  virtual unsigned outputWidth() const = 0;
  virtual unsigned outputHeight() const = 0;
  virtual unsigned outputPitch() const = 0;
  virtual std::uint64_t frameId() const = 0;
};

}  // namespace f8::implayer
