#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace f8::cppsdk {
class VideoSharedMemorySink;
}

namespace f8::screencap {

// Windows Graphics Capture (WinRT) backend.
//
// Notes:
// - display/window are captured as a GraphicsCaptureItem (not desktop region), so overlays won't be captured like gdigrab.
// - region is implemented by capturing a monitor and cropping.
class Win32WgcCapture final {
 public:
  using OnFrameFn = std::function<void(std::uint64_t frame_id, std::int64_t ts_ms)>;
  using OnRunningFn = std::function<void(bool running)>;
  using OnErrorFn = std::function<void(std::string err)>;

  struct Config {
    std::string mode = "display";  // display|window|region
    double fps = 30.0;
    int display_id = 0;
    std::string window_id;   // win32:hwnd:0x...
    std::string region_csv;  // x,y,w,h in virtual desktop coords
    std::string scale_csv;   // w,h (optional; 0,0 disables)
  };

  Win32WgcCapture(std::string service_id, std::shared_ptr<f8::cppsdk::VideoSharedMemorySink> sink);
  ~Win32WgcCapture();

  void configure(std::string mode, double fps, int display_id, std::string window_id, std::string region_csv,
                 std::string scale_csv);
  void set_active(bool active) { active_.store(active, std::memory_order_release); }
  void restart();
  void tick();

  void set_on_frame(OnFrameFn fn) { on_frame_ = std::move(fn); }
  void set_on_running(OnRunningFn fn) { on_running_ = std::move(fn); }
  void set_on_error(OnErrorFn fn) { on_error_ = std::move(fn); }

 private:
  void close_capture();
  bool open_capture(std::string& err);
  void pump_capture();
  void set_error(std::string err);

  static void scale_bgra_bilinear(const std::uint8_t* src, int src_w, int src_h, int src_stride, std::uint8_t* dst,
                                  int dst_w, int dst_h, int dst_stride);

  std::string service_id_;
  std::shared_ptr<f8::cppsdk::VideoSharedMemorySink> sink_;

  std::atomic<bool> active_{true};
  std::atomic<bool> want_restart_{false};
  std::atomic<bool> opened_{false};

  std::mutex cfg_mu_;
  Config cfg_;

  OnFrameFn on_frame_;
  OnRunningFn on_running_;
  OnErrorFn on_error_;

  std::uint64_t frame_id_ = 0;
  std::int64_t last_write_ts_ms_ = 0;

  // Windows-only runtime state (kept opaque in header).
  void* rt_ = nullptr;

  std::vector<std::uint8_t> out_bgra_;
};

}  // namespace f8::screencap
