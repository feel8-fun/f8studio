#pragma once

#include <cstdint>
#include <string>

#include "f8cppsdk/shm/naming.h"

namespace f8::cppsdk::shm {

// Cross-platform helper for waiting on a SHM "new frame" signal.
// - Windows: uses a named event `shmName_evt` (created by the writer).
// - Others: no-op (polling expected).
class FrameEventWaiter final {
 public:
  FrameEventWaiter() = default;
  ~FrameEventWaiter();
  FrameEventWaiter(const FrameEventWaiter&) = delete;
  FrameEventWaiter& operator=(const FrameEventWaiter&) = delete;

  bool open_for_shm(const std::string& shm_name);
  void close();

  // Returns true if signaled, false on timeout or unsupported.
  bool wait(std::uint32_t timeout_ms) const;

 private:
  void* handle_ = nullptr;
};

}  // namespace f8::cppsdk::shm

