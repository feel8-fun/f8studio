#pragma once

#include <functional>
#include <string>

#include "x11_capture_sources.h"

namespace f8::screencap::x11 {

struct PickRegionResult {
  bool ok = false;
  RectI rect;
  std::string error;
};

class X11Picker final {
 public:
  using PickRegionCallback = std::function<void(PickRegionResult)>;

  static void pick_region_async(PickRegionCallback cb);
};

}  // namespace f8::screencap::x11

