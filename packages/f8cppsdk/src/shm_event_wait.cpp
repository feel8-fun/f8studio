#include "f8cppsdk/shm/event_wait.h"

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#endif

namespace f8::cppsdk::shm {

FrameEventWaiter::~FrameEventWaiter() { close(); }

bool FrameEventWaiter::open_for_shm(const std::string& shm_name) {
  close();
#if defined(_WIN32)
  const auto ev_name = frame_event_name(shm_name);
  const std::wstring wname(ev_name.begin(), ev_name.end());
  HANDLE h = OpenEventW(SYNCHRONIZE, FALSE, wname.c_str());
  if (!h) return false;
  handle_ = h;
  return true;
#else
  (void)shm_name;
  return false;
#endif
}

void FrameEventWaiter::close() {
#if defined(_WIN32)
  if (handle_) {
    CloseHandle(static_cast<HANDLE>(handle_));
    handle_ = nullptr;
  }
#else
  handle_ = nullptr;
#endif
}

bool FrameEventWaiter::wait(std::uint32_t timeout_ms) const {
#if defined(_WIN32)
  if (!handle_) return false;
  const DWORD rc = WaitForSingleObject(static_cast<HANDLE>(handle_), static_cast<DWORD>(timeout_ms));
  return rc == WAIT_OBJECT_0;
#else
  (void)timeout_ms;
  return false;
#endif
}

}  // namespace f8::cppsdk::shm

