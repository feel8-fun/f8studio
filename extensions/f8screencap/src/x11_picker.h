#pragma once

#include <functional>
#include <string>

#include "x11_capture_sources.h"

namespace f8::screencap::x11 {

struct PickDisplayResult {
  bool ok = false;
  int display_id = 0;
  RectI rect;
  std::string name;
  bool primary = false;
  std::string error;
};

struct PickWindowResult {
  bool ok = false;
  std::string window_id;  // x11:win:0x...
  RectI rect;
  std::string title;
  std::uint32_t pid = 0;
  std::string error;
};

struct PickRegionResult {
  bool ok = false;
  RectI rect;
  std::string error;
};

class X11Picker final {
 public:
  using PickDisplayCallback = std::function<void(PickDisplayResult)>;
  using PickWindowCallback = std::function<void(PickWindowResult)>;
  using PickRegionCallback = std::function<void(PickRegionResult)>;

  static void pick_display_async(PickDisplayCallback cb);
  static void pick_window_async(PickWindowCallback cb);
  static void pick_region_async(PickRegionCallback cb);
};

}  // namespace f8::screencap::x11
