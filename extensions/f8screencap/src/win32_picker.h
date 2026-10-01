#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "win32_capture_sources.h"

namespace f8::screencap::win32 {

struct PickDisplayResult {
  bool ok = false;
  int display_id = -1;
  std::string err;
};

struct PickWindowResult {
  bool ok = false;
  std::string window_id;  // win32:hwnd:0x...
  RectI rect{};
  std::string title;
  std::uint32_t pid = 0;
  std::string err;
};

struct PickRegionResult {
  bool ok = false;
  RectI rect{};
  std::string err;
};

// Windows-only interactive pickers:
// - Display: hover highlights monitor; LMB confirms; RMB/ESC cancels.
// - Window: hover highlights window; LMB confirms; RMB/ESC cancels.
// - Region: click-drag to draw; LMB up confirms; RMB/ESC cancels.
class Win32Picker final {
 public:
  using OnPickDisplay = std::function<void(PickDisplayResult)>;
  using OnPickWindow = std::function<void(PickWindowResult)>;
  using OnPickRegion = std::function<void(PickRegionResult)>;

  static void pick_display_async(OnPickDisplay cb);
  static void pick_window_async(OnPickWindow cb);
  static void pick_region_async(OnPickRegion cb);
};

}  // namespace f8::screencap::win32

