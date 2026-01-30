#include "linux_x11_capture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

#include <spdlog/spdlog.h>

#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/video_shared_memory_sink.h"
#include "x11_capture_sources.h"

#if defined(__linux__) && !defined(_WIN32)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace f8::screencap {

using f8::cppsdk::now_ms;

namespace {

struct X11Runtime {
#if defined(__linux__) && !defined(_WIN32)
  Display* dpy = nullptr;
  int screen = 0;
  Window root = 0;
  bool have_window = false;
  Window window = 0;
#endif

  int src_x = 0;
  int src_y = 0;
  int src_w = 0;
  int src_h = 0;
};

std::int64_t now_ms_steady() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

struct MaskInfo {
  std::uint32_t mask = 0;
  int shift = 0;
  int bits = 0;
};

MaskInfo mask_info(std::uint32_t mask) {
  MaskInfo mi;
  mi.mask = mask;
  if (mask == 0) return mi;
  int shift = 0;
  while (((mask >> shift) & 1u) == 0u && shift < 32) ++shift;
  mi.shift = shift;
  std::uint32_t m = mask >> shift;
  int bits = 0;
  while ((m & 1u) != 0u) {
    ++bits;
    m >>= 1u;
  }
  mi.bits = bits;
  return mi;
}

std::uint8_t scale_to_u8(std::uint32_t v, int bits) {
  if (bits <= 0) return 0;
  if (bits >= 8) return static_cast<std::uint8_t>(std::clamp<std::uint32_t>(v >> (bits - 8), 0u, 255u));
  const std::uint32_t maxv = (1u << bits) - 1u;
  return static_cast<std::uint8_t>(std::clamp<std::uint32_t>((v * 255u) / std::max(1u, maxv), 0u, 255u));
}

}  // namespace

LinuxX11Capture::LinuxX11Capture(std::string service_id, std::shared_ptr<f8::cppsdk::VideoSharedMemorySink> sink)
    : service_id_(std::move(service_id)), sink_(std::move(sink)) {}

LinuxX11Capture::~LinuxX11Capture() { close_capture(); }

void LinuxX11Capture::configure(std::string mode, double fps, int display_id, std::string window_id, std::string region_csv,
                                std::string scale_csv) {
  {
    std::lock_guard<std::mutex> lock(cfg_mu_);
    cfg_.mode = std::move(mode);
    cfg_.fps = fps;
    cfg_.display_id = display_id;
    cfg_.window_id = std::move(window_id);
    cfg_.region_csv = std::move(region_csv);
    cfg_.scale_csv = std::move(scale_csv);
  }
  want_restart_.store(true, std::memory_order_release);
}

void LinuxX11Capture::restart() { want_restart_.store(true, std::memory_order_release); }

void LinuxX11Capture::tick() {
  const bool active = active_.load(std::memory_order_acquire);
  if (!active) {
    if (opened_.exchange(false, std::memory_order_acq_rel)) close_capture();
    return;
  }

  if (want_restart_.exchange(false, std::memory_order_acq_rel) || !opened_.load(std::memory_order_acquire)) {
    close_capture();
    std::string err;
    if (!open_capture(err)) {
      set_error(err.empty() ? "open_capture failed" : err);
      return;
    }
  }

  pump_capture();
}

void LinuxX11Capture::close_capture() {
  opened_.store(false, std::memory_order_release);
  if (on_running_) on_running_(false);

  auto* rt = reinterpret_cast<X11Runtime*>(rt_);
  if (!rt) return;
#if defined(__linux__) && !defined(_WIN32)
  if (rt->dpy) {
    XCloseDisplay(rt->dpy);
    rt->dpy = nullptr;
  }
#endif
  delete rt;
  rt_ = nullptr;
}

bool LinuxX11Capture::open_capture(std::string& err) {
  err.clear();
  if (!sink_) {
    err = "sink not set";
    return false;
  }

#if !defined(__linux__) || defined(_WIN32)
  err = "linux x11 capture not supported on this platform";
  return false;
#else
  const Config cfg = [&]() {
    std::lock_guard<std::mutex> lock(cfg_mu_);
    return cfg_;
  }();

  auto rt = std::make_unique<X11Runtime>();
  rt->dpy = XOpenDisplay(nullptr);
  if (!rt->dpy) {
    err = "XOpenDisplay failed (is DISPLAY set?)";
    return false;
  }

  const int screens = ScreenCount(rt->dpy);
  rt->screen = std::clamp(cfg.display_id, 0, std::max(0, screens - 1));
  rt->root = RootWindow(rt->dpy, rt->screen);
  const int root_w = DisplayWidth(rt->dpy, rt->screen);
  const int root_h = DisplayHeight(rt->dpy, rt->screen);

  int src_w = 0;
  int src_h = 0;
  int src_x = 0;
  int src_y = 0;

  if (cfg.mode == "display") {
    std::string derr;
    const auto displays = x11::enumerate_displays(derr);
    const x11::DisplayInfo* sel = nullptr;
    for (const auto& d : displays) {
      if (d.id == cfg.display_id) {
        sel = &d;
        break;
      }
    }
    if (sel) {
      src_x = sel->rect.x;
      src_y = sel->rect.y;
      src_w = sel->rect.w;
      src_h = sel->rect.h;
    } else {
      src_x = 0;
      src_y = 0;
      src_w = root_w;
      src_h = root_h;
    }
  } else if (cfg.mode == "window") {
    std::uint64_t xid = 0;
    std::string perr;
    if (!x11::try_parse_window_id(cfg.window_id, xid, perr)) {
      err = "invalid window_id: " + perr;
      return false;
    }
    rt->window = static_cast<Window>(xid);
    rt->have_window = true;

    XWindowAttributes wa{};
    if (!XGetWindowAttributes(rt->dpy, rt->window, &wa)) {
      err = "XGetWindowAttributes failed for window";
      return false;
    }
    src_x = 0;
    src_y = 0;
    src_w = wa.width;
    src_h = wa.height;
  } else if (cfg.mode == "region") {
    x11::RectI rc{};
    std::string perr;
    if (!x11::try_parse_csv_rect(cfg.region_csv, rc, perr)) {
      err = "invalid region: " + perr;
      return false;
    }
    src_x = std::clamp(rc.x, 0, std::max(0, root_w - 1));
    src_y = std::clamp(rc.y, 0, std::max(0, root_h - 1));
    src_w = std::clamp(rc.w, 1, std::max(1, root_w - src_x));
    src_h = std::clamp(rc.h, 1, std::max(1, root_h - src_y));
  } else {
    err = "invalid mode";
    return false;
  }

  if (src_w <= 0 || src_h <= 0) {
    err = "invalid capture dimensions";
    return false;
  }

  int scale_w = 0;
  int scale_h = 0;
  {
    std::string perr;
    if (!x11::try_parse_csv_size(cfg.scale_csv, scale_w, scale_h, perr)) {
      err = "invalid scale: " + perr;
      return false;
    }
  }
  const bool want_scale = (scale_w > 0 && scale_h > 0);
  const int out_w = want_scale ? scale_w : src_w;
  const int out_h = want_scale ? scale_h : src_h;
  if (!sink_->ensureConfiguration(static_cast<unsigned>(out_w), static_cast<unsigned>(out_h))) {
    err = "shm ensureConfiguration failed";
    return false;
  }

  rt->src_x = src_x;
  rt->src_y = src_y;
  rt->src_w = src_w;
  rt->src_h = src_h;

  frame_id_ = 0;
  last_write_ts_ms_ = 0;
  rt_ = rt.release();
  opened_.store(true, std::memory_order_release);
  set_error(std::string{});
  if (on_running_) on_running_(true);

  // Placeholder frame.
  {
    const int ow = static_cast<int>(sink_->outputWidth());
    const int oh = static_cast<int>(sink_->outputHeight());
    const int ostride = ow * 4;
    if (ow > 0 && oh > 0) {
      out_bgra_.assign(static_cast<std::size_t>(ostride) * oh, 0);
      if (sink_->writeFrame(out_bgra_.data(), static_cast<unsigned>(ostride))) {
        if (on_frame_) on_frame_(sink_->frameId(), now_ms());
      }
    }
  }

  return true;
#endif
}

void LinuxX11Capture::pump_capture() {
  auto* rt = reinterpret_cast<X11Runtime*>(rt_);
  if (!rt || !sink_) return;

#if !defined(__linux__) || defined(_WIN32)
  return;
#else
  const Config cfg = [&]() {
    std::lock_guard<std::mutex> lock(cfg_mu_);
    return cfg_;
  }();

  const double fps = (cfg.fps > 0.0) ? cfg.fps : 30.0;
  const std::int64_t min_interval = static_cast<std::int64_t>(1000.0 / fps);

  const std::int64_t t0 = now_ms_steady();
  if (last_write_ts_ms_ != 0 && t0 - last_write_ts_ms_ < min_interval) {
    return;
  }

  const int src_w = rt->src_w;
  const int src_h = rt->src_h;
  if (src_w <= 0 || src_h <= 0) return;

  XImage* img = nullptr;
  if (rt->have_window) {
    img = XGetImage(rt->dpy, rt->window, 0, 0, static_cast<unsigned>(src_w), static_cast<unsigned>(src_h), AllPlanes, ZPixmap);
  } else {
    img = XGetImage(rt->dpy, rt->root, rt->src_x, rt->src_y, static_cast<unsigned>(src_w), static_cast<unsigned>(src_h), AllPlanes,
                    ZPixmap);
  }

  if (!img || !img->data) {
    set_error("XGetImage failed");
    want_restart_.store(true, std::memory_order_release);
    if (img) XDestroyImage(img);
    return;
  }

  const int out_w = static_cast<int>(sink_->outputWidth());
  const int out_h = static_cast<int>(sink_->outputHeight());
  const int out_stride = out_w * 4;
  if (out_w <= 0 || out_h <= 0) {
    XDestroyImage(img);
    return;
  }

  if (out_bgra_.size() != static_cast<std::size_t>(out_stride) * out_h) {
    out_bgra_.resize(static_cast<std::size_t>(out_stride) * out_h);
  }

  // Convert XImage to src BGRA.
  const int bpp = img->bits_per_pixel;
  const int bytes_per_line = img->bytes_per_line;
  if (bpp != 32 && bpp != 24) {
    XDestroyImage(img);
    set_error("unsupported XImage bits_per_pixel");
    return;
  }

  const auto rmi = mask_info(static_cast<std::uint32_t>(img->red_mask));
  const auto gmi = mask_info(static_cast<std::uint32_t>(img->green_mask));
  const auto bmi = mask_info(static_cast<std::uint32_t>(img->blue_mask));
  src_bgra_.resize(static_cast<std::size_t>(src_w) * src_h * 4);

  const bool lsb_first = (img->byte_order == LSBFirst);
  const auto read_u32 = [&](const std::uint8_t* p) -> std::uint32_t {
    if (lsb_first) {
      return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8u) | (static_cast<std::uint32_t>(p[2]) << 16u) |
             (static_cast<std::uint32_t>(p[3]) << 24u);
    }
    return static_cast<std::uint32_t>(p[3]) | (static_cast<std::uint32_t>(p[2]) << 8u) | (static_cast<std::uint32_t>(p[1]) << 16u) |
           (static_cast<std::uint32_t>(p[0]) << 24u);
  };
  const auto read_u24 = [&](const std::uint8_t* p) -> std::uint32_t {
    if (lsb_first) {
      return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8u) | (static_cast<std::uint32_t>(p[2]) << 16u);
    }
    return static_cast<std::uint32_t>(p[2]) | (static_cast<std::uint32_t>(p[1]) << 8u) | (static_cast<std::uint32_t>(p[0]) << 16u);
  };

  const auto* src = reinterpret_cast<const std::uint8_t*>(img->data);
  for (int y = 0; y < src_h; ++y) {
    const std::uint8_t* row = src + static_cast<std::size_t>(y) * bytes_per_line;
    std::uint8_t* out = src_bgra_.data() + static_cast<std::size_t>(y) * src_w * 4;
    for (int x = 0; x < src_w; ++x) {
      std::uint32_t px = 0;
      if (bpp == 32) {
        px = read_u32(row + static_cast<std::size_t>(x) * 4);
      } else {
        px = read_u24(row + static_cast<std::size_t>(x) * 3);
      }

      const std::uint32_t rv = (rmi.mask != 0) ? ((px & rmi.mask) >> rmi.shift) : 0u;
      const std::uint32_t gv = (gmi.mask != 0) ? ((px & gmi.mask) >> gmi.shift) : 0u;
      const std::uint32_t bv = (bmi.mask != 0) ? ((px & bmi.mask) >> bmi.shift) : 0u;

      out[x * 4 + 0] = scale_to_u8(bv, bmi.bits);
      out[x * 4 + 1] = scale_to_u8(gv, gmi.bits);
      out[x * 4 + 2] = scale_to_u8(rv, rmi.bits);
      out[x * 4 + 3] = 255;
    }
  }

  XDestroyImage(img);

  const int src_stride = src_w * 4;
  if (out_w == src_w && out_h == src_h) {
    std::memcpy(out_bgra_.data(), src_bgra_.data(), src_bgra_.size());
  } else {
    scale_bgra_bilinear(src_bgra_.data(), src_w, src_h, src_stride, out_bgra_.data(), out_w, out_h, out_stride);
  }

  if (!sink_->writeFrame(out_bgra_.data(), static_cast<unsigned>(out_stride))) {
    set_error("shm writeFrame failed");
    return;
  }

  last_write_ts_ms_ = t0;
  const std::uint64_t fid = ++frame_id_;
  if (on_frame_) on_frame_(fid, now_ms());
#endif
}

void LinuxX11Capture::set_error(std::string err) {
  if (err.empty()) {
    if (on_error_) on_error_(std::string{});
    return;
  }
  spdlog::warn("x11 capture error: {}", err);
  if (on_error_) on_error_(std::move(err));
}

void LinuxX11Capture::scale_bgra_bilinear(const std::uint8_t* src, int src_w, int src_h, int src_stride, std::uint8_t* dst,
                                          int dst_w, int dst_h, int dst_stride) {
  if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;
  const float sx = static_cast<float>(src_w) / static_cast<float>(dst_w);
  const float sy = static_cast<float>(src_h) / static_cast<float>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    const float fy = (static_cast<float>(y) + 0.5f) * sy - 0.5f;
    int y0 = static_cast<int>(std::floor(fy));
    int y1 = y0 + 1;
    const float wy = fy - static_cast<float>(y0);
    y0 = std::clamp(y0, 0, src_h - 1);
    y1 = std::clamp(y1, 0, src_h - 1);

    auto* out = dst + static_cast<std::size_t>(y) * dst_stride;
    const auto* row0 = src + static_cast<std::size_t>(y0) * src_stride;
    const auto* row1 = src + static_cast<std::size_t>(y1) * src_stride;

    for (int x = 0; x < dst_w; ++x) {
      const float fx = (static_cast<float>(x) + 0.5f) * sx - 0.5f;
      int x0 = static_cast<int>(std::floor(fx));
      int x1 = x0 + 1;
      const float wx = fx - static_cast<float>(x0);
      x0 = std::clamp(x0, 0, src_w - 1);
      x1 = std::clamp(x1, 0, src_w - 1);

      const std::uint8_t* p00 = row0 + x0 * 4;
      const std::uint8_t* p10 = row0 + x1 * 4;
      const std::uint8_t* p01 = row1 + x0 * 4;
      const std::uint8_t* p11 = row1 + x1 * 4;

      for (int c = 0; c < 4; ++c) {
        const float v0 = static_cast<float>(p00[c]) * (1.0f - wx) + static_cast<float>(p10[c]) * wx;
        const float v1 = static_cast<float>(p01[c]) * (1.0f - wx) + static_cast<float>(p11[c]) * wx;
        const float v = v0 * (1.0f - wy) + v1 * wy;
        out[x * 4 + c] = static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f));
      }
    }
  }
}

}  // namespace f8::screencap
