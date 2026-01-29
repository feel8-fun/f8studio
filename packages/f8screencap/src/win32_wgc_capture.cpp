#include "win32_wgc_capture.h"

#if defined(_WIN32)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <Windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#include <Windows.Graphics.Capture.Interop.h>
#include <Windows.Graphics.DirectX.Direct3D11.interop.h>

#include <spdlog/spdlog.h>

#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/video_shared_memory_sink.h"
#include "win32_capture_sources.h"

namespace f8::screencap {

using namespace winrt;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

namespace {

std::int64_t now_ms_steady() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

struct SizeI {
  int w = 0;
  int h = 0;
};

bool parse_scale_csv(const std::string& csv, SizeI& out, std::string& err) {
  out = {};
  if (csv.empty()) return true;
  int w = 0, h = 0;
  if (!win32::try_parse_csv_size(csv, w, h, err)) return false;
  out.w = w;
  out.h = h;
  return true;
}

bool parse_region_csv(const std::string& csv, win32::RectI& out, std::string& err) {
  out = {};
  if (csv.empty()) {
    err = "region is empty";
    return false;
  }
  return win32::try_parse_csv_rect(csv, out, err);
}

GraphicsCaptureItem make_item_for_window(HWND hwnd, std::string& err) {
  err.clear();
  GraphicsCaptureItem item{nullptr};
  if (!IsWindow(hwnd)) {
    err = "hwnd is not a window";
    return item;
  }
  auto interop = get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
  HRESULT hr = interop->CreateForWindow(hwnd, guid_of<GraphicsCaptureItem>(), reinterpret_cast<void**>(put_abi(item)));
  if (FAILED(hr)) {
    err = "CreateForWindow failed";
  }
  return item;
}

GraphicsCaptureItem make_item_for_monitor(HMONITOR mon, std::string& err) {
  err.clear();
  GraphicsCaptureItem item{nullptr};
  if (!mon) {
    err = "monitor is null";
    return item;
  }
  auto interop = get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
  HRESULT hr = interop->CreateForMonitor(mon, guid_of<GraphicsCaptureItem>(), reinterpret_cast<void**>(put_abi(item)));
  if (FAILED(hr)) {
    err = "CreateForMonitor failed";
  }
  return item;
}

IDirect3DDevice make_direct3d_device(com_ptr<ID3D11Device> const& dev) {
  com_ptr<IDXGIDevice> dxgi;
  dev.as(dxgi);

  com_ptr<IInspectable> insp;
  winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), insp.put()));
  return insp.as<IDirect3DDevice>();
}

struct Crop {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

Crop clamp_crop(Crop c, int src_w, int src_h) {
  if (src_w <= 0 || src_h <= 0) return Crop{};
  c.x = std::max(0, c.x);
  c.y = std::max(0, c.y);
  c.w = std::max(0, std::min(c.w, src_w - c.x));
  c.h = std::max(0, std::min(c.h, src_h - c.y));
  return c;
}

}  // namespace

struct Win32WgcRuntime final {
  com_ptr<ID3D11Device> d3d;
  com_ptr<ID3D11DeviceContext> ctx;

  IDirect3DDevice d3d_rt{nullptr};
  GraphicsCaptureItem item{nullptr};
  Direct3D11CaptureFramePool frame_pool{nullptr};
  GraphicsCaptureSession session{nullptr};
  winrt::event_token frame_token{};

  SizeInt32 last_size{0, 0};
  std::int64_t last_try_ms = 0;

  // For region crop: monitor rect in virtual desktop coordinates.
  win32::RectI src_virtual_rect{};
  bool have_src_virtual_rect = false;

  // Crop in source pixels.
  Crop crop{};
  bool have_crop = false;
};

Win32WgcCapture::Win32WgcCapture(std::string service_id, std::shared_ptr<f8::cppsdk::VideoSharedMemorySink> sink)
    : service_id_(std::move(service_id)), sink_(std::move(sink)) {
  winrt::init_apartment(winrt::apartment_type::multi_threaded);
}

Win32WgcCapture::~Win32WgcCapture() {
  close_capture();
}

void Win32WgcCapture::configure(std::string mode, double fps, int display_id, std::string window_id, std::string region_csv,
                                std::string scale_csv) {
  std::lock_guard<std::mutex> lock(cfg_mu_);
  cfg_.mode = std::move(mode);
  cfg_.fps = fps;
  cfg_.display_id = display_id;
  cfg_.window_id = std::move(window_id);
  cfg_.region_csv = std::move(region_csv);
  cfg_.scale_csv = std::move(scale_csv);
}

void Win32WgcCapture::restart() {
  want_restart_.store(true, std::memory_order_release);
}

void Win32WgcCapture::tick() {
  const bool active = active_.load(std::memory_order_acquire);

  if (!active) {
    if (opened_.exchange(false, std::memory_order_acq_rel)) {
      close_capture();
      if (on_running_) on_running_(false);
    }
    return;
  }

  if (want_restart_.exchange(false, std::memory_order_acq_rel)) {
    close_capture();
    opened_.store(false, std::memory_order_release);
    if (on_running_) on_running_(false);
  }

  if (!opened_.load(std::memory_order_acquire)) {
    std::string err;
    if (!open_capture(err)) {
      set_error(std::move(err));
      return;
    }
    opened_.store(true, std::memory_order_release);
    if (on_running_) on_running_(true);
  }

  pump_capture();
}

void Win32WgcCapture::close_capture() {
  auto* rt = reinterpret_cast<Win32WgcRuntime*>(rt_);
  if (!rt) return;

  try {
    if (rt->frame_pool && rt->frame_token.value) {
      rt->frame_pool.FrameArrived(rt->frame_token);
      rt->frame_token = {};
    }
  } catch (...) {}

  try {
    if (rt->session) rt->session.Close();
  } catch (...) {}
  try {
    if (rt->frame_pool) rt->frame_pool.Close();
  } catch (...) {}

  rt->session = nullptr;
  rt->frame_pool = nullptr;
  rt->item = nullptr;
  rt->d3d_rt = nullptr;
  rt->ctx = nullptr;
  rt->d3d = nullptr;

  delete rt;
  rt_ = nullptr;
}

bool Win32WgcCapture::open_capture(std::string& err) {
  err.clear();
  close_capture();

  if (!GraphicsCaptureSession::IsSupported()) {
    err = "Windows Graphics Capture not supported on this OS";
    return false;
  }
  if (!sink_) {
    err = "video shm sink not initialized";
    return false;
  }

  Config cfg;
  {
    std::lock_guard<std::mutex> lock(cfg_mu_);
    cfg = cfg_;
  }
  if (cfg.fps <= 0.0) cfg.fps = 30.0;

  auto rt = std::make_unique<Win32WgcRuntime>();

  // D3D11 device (BGRA required for interop surfaces).
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
  D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL fl_out{};
  com_ptr<ID3D11Device> dev;
  com_ptr<ID3D11DeviceContext> ctx;
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, static_cast<UINT>(std::size(levels)),
                                 D3D11_SDK_VERSION, dev.put(), &fl_out, ctx.put());
  if (FAILED(hr)) {
    err = "D3D11CreateDevice failed";
    return false;
  }
  rt->d3d = std::move(dev);
  rt->ctx = std::move(ctx);
  rt->d3d_rt = make_direct3d_device(rt->d3d);

  // Select capture item & crop rules.
  if (cfg.mode == "window") {
    std::uintptr_t hwnd_u = 0;
    if (!win32::try_parse_window_id(cfg.window_id, hwnd_u, err)) return false;
    rt->item = make_item_for_window(reinterpret_cast<HWND>(hwnd_u), err);
    if (!rt->item) return false;
    rt->have_crop = false;
  } else if (cfg.mode == "display") {
    std::uintptr_t hmon_u = 0;
    if (!win32::try_get_monitor_handle(cfg.display_id, hmon_u, err)) return false;
    rt->item = make_item_for_monitor(reinterpret_cast<HMONITOR>(hmon_u), err);
    if (!rt->item) return false;
    // Full monitor (no crop).
    rt->have_crop = false;
  } else if (cfg.mode == "region") {
    win32::RectI region{};
    if (!parse_region_csv(cfg.region_csv, region, err)) return false;

    // Pick the monitor that contains the region center (best-effort).
    POINT pt{};
    pt.x = region.x + region.w / 2;
    pt.y = region.y + region.h / 2;
    HMONITOR mon = MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
    if (!mon) {
      err = "MonitorFromPoint failed";
      return false;
    }

    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi)) {
      err = "GetMonitorInfoW failed";
      return false;
    }
    const win32::RectI mon_rect{mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                                mi.rcMonitor.bottom - mi.rcMonitor.top};
    rt->src_virtual_rect = mon_rect;
    rt->have_src_virtual_rect = true;

    // Crop relative to monitor capture.
    const int cx0 = std::max(region.x, mon_rect.x);
    const int cy0 = std::max(region.y, mon_rect.y);
    const int cx1 = std::min(region.x + region.w, mon_rect.x + mon_rect.w);
    const int cy1 = std::min(region.y + region.h, mon_rect.y + mon_rect.h);
    Crop crop;
    crop.x = cx0 - mon_rect.x;
    crop.y = cy0 - mon_rect.y;
    crop.w = std::max(0, cx1 - cx0);
    crop.h = std::max(0, cy1 - cy0);
    if (crop.w <= 1 || crop.h <= 1) {
      err = "region does not intersect monitor";
      return false;
    }
    rt->crop = crop;
    rt->have_crop = true;

    rt->item = make_item_for_monitor(mon, err);
    if (!rt->item) return false;
  } else {
    err = "unknown mode: " + cfg.mode;
    return false;
  }

  const auto size = rt->item.Size();
  rt->last_size = size;
  const int src_w = std::max(1, size.Width);
  const int src_h = std::max(1, size.Height);
  Crop crop = rt->have_crop ? clamp_crop(rt->crop, src_w, src_h) : Crop{0, 0, src_w, src_h};
  if (crop.w <= 0 || crop.h <= 0) {
    err = "invalid capture size";
    return false;
  }

  SizeI user_scale{};
  if (!parse_scale_csv(cfg.scale_csv, user_scale, err)) return false;
  int req_w = (user_scale.w > 0) ? user_scale.w : crop.w;
  int req_h = (user_scale.h > 0) ? user_scale.h : crop.h;
  if (req_w <= 0 || req_h <= 0) {
    err = "invalid output size";
    return false;
  }
  if (!sink_->ensureConfiguration(static_cast<unsigned>(req_w), static_cast<unsigned>(req_h))) {
    err = "failed to configure shm dimensions (capacity too small?)";
    return false;
  }
  const int out_w = static_cast<int>(sink_->outputWidth());
  const int out_h = static_cast<int>(sink_->outputHeight());
  out_bgra_.resize(static_cast<std::size_t>(out_w) * out_h * 4);

  rt->frame_pool = Direct3D11CaptureFramePool::Create(rt->d3d_rt, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size);
  rt->session = rt->frame_pool.CreateCaptureSession(rt->item);

  // Optional signal; pump still falls back to polling so viewer won't "hang" if the event doesn't fire.
  rt->frame_token = rt->frame_pool.FrameArrived([rt_raw = rt.get()](auto&&, auto&&) {
    if (rt_raw) rt_raw->last_try_ms = 0;  // allow immediate TryGetNextFrame
  });

  try {
    rt->session.StartCapture();
  } catch (...) {
    err = "StartCapture failed (privacy settings?)";
    return false;
  }

  frame_id_ = 0;
  last_write_ts_ms_ = 0;
  rt_ = rt.release();
  set_error(std::string{});

  // Write one placeholder frame so consumers (e.g. videoshm_viewer.py) can see valid dimensions immediately
  // even if no frames arrive yet.
  {
    const int out_w = static_cast<int>(sink_->outputWidth());
    const int out_h = static_cast<int>(sink_->outputHeight());
    const int out_stride = out_w * 4;
    if (out_w > 0 && out_h > 0 && out_bgra_.size() == static_cast<std::size_t>(out_stride) * out_h) {
      std::memset(out_bgra_.data(), 0, out_bgra_.size());
      if (sink_->writeFrame(out_bgra_.data(), static_cast<unsigned>(out_stride))) {
        if (on_frame_) on_frame_(sink_->frameId(), f8::cppsdk::now_ms());
      }
    }
  }
  return true;
}

void Win32WgcCapture::pump_capture() {
  auto* rt = reinterpret_cast<Win32WgcRuntime*>(rt_);
  if (!rt || !rt->frame_pool || !rt->ctx || !sink_) return;

  const Config cfg = [&]() {
    std::lock_guard<std::mutex> lock(cfg_mu_);
    return cfg_;
  }();

  const double fps = (cfg.fps > 0.0) ? cfg.fps : 30.0;
  const std::int64_t min_interval = static_cast<std::int64_t>(1000.0 / fps);

  const std::int64_t now = now_ms_steady();
  // Poll fallback: TryGetNextFrame can return null when no frame is available.
  // Limit the polling rate to avoid pegging CPU when capture is blocked.
  if (rt->last_try_ms != 0 && now - rt->last_try_ms < 10) {
    return;
  }
  rt->last_try_ms = now;

  Direct3D11CaptureFrame frame{nullptr};
  try {
    frame = rt->frame_pool.TryGetNextFrame();
  } catch (...) {
    set_error("TryGetNextFrame failed");
    want_restart_.store(true, std::memory_order_release);
    return;
  }
  if (!frame) return;

  // Handle resize.
  const auto size = frame.ContentSize();
  if (size.Width != rt->last_size.Width || size.Height != rt->last_size.Height) {
    rt->last_size = size;
    try {
      rt->frame_pool.Recreate(rt->d3d_rt, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size);
    } catch (...) {
      set_error("frame_pool.Recreate failed");
      want_restart_.store(true, std::memory_order_release);
      return;
    }
  }

  const std::int64_t t0 = now_ms_steady();
  if (last_write_ts_ms_ != 0 && t0 - last_write_ts_ms_ < min_interval) {
    return;
  }

  // Get D3D11 texture from surface.
  com_ptr<ID3D11Texture2D> src_tex;
  try {
    auto surface = frame.Surface();
    auto access = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D), src_tex.put_void()));
  } catch (...) {
    set_error("GetInterface(ID3D11Texture2D) failed");
    return;
  }

  D3D11_TEXTURE2D_DESC desc{};
  src_tex->GetDesc(&desc);
  const int src_w = static_cast<int>(desc.Width);
  const int src_h = static_cast<int>(desc.Height);
  if (src_w <= 0 || src_h <= 0) return;

  Crop crop = rt->have_crop ? clamp_crop(rt->crop, src_w, src_h) : Crop{0, 0, src_w, src_h};
  if (crop.w <= 0 || crop.h <= 0) return;

  // Create staging texture for the cropped region.
  D3D11_TEXTURE2D_DESC st{};
  st.Width = static_cast<UINT>(crop.w);
  st.Height = static_cast<UINT>(crop.h);
  st.MipLevels = 1;
  st.ArraySize = 1;
  st.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  st.SampleDesc.Count = 1;
  st.Usage = D3D11_USAGE_STAGING;
  st.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

  com_ptr<ID3D11Texture2D> staging;
  HRESULT hr = rt->d3d->CreateTexture2D(&st, nullptr, staging.put());
  if (FAILED(hr) || !staging) {
    set_error("CreateTexture2D(staging) failed");
    return;
  }

  D3D11_BOX box{};
  box.left = static_cast<UINT>(crop.x);
  box.top = static_cast<UINT>(crop.y);
  box.front = 0;
  box.right = static_cast<UINT>(crop.x + crop.w);
  box.bottom = static_cast<UINT>(crop.y + crop.h);
  box.back = 1;
  rt->ctx->CopySubresourceRegion(staging.get(), 0, 0, 0, 0, src_tex.get(), 0, &box);

  D3D11_MAPPED_SUBRESOURCE mapped{};
  hr = rt->ctx->Map(staging.get(), 0, D3D11_MAP_READ, 0, &mapped);
  if (FAILED(hr) || !mapped.pData) {
    set_error("Map(staging) failed");
    return;
  }

  const auto unmap = [&]() { rt->ctx->Unmap(staging.get(), 0); };

  // Prepare output.
  const int out_w = static_cast<int>(sink_->outputWidth());
  const int out_h = static_cast<int>(sink_->outputHeight());
  const int out_stride = out_w * 4;
  if (out_w <= 0 || out_h <= 0) {
    unmap();
    return;
  }
  if (out_bgra_.size() != static_cast<std::size_t>(out_stride) * out_h) {
    out_bgra_.resize(static_cast<std::size_t>(out_stride) * out_h);
  }

  const auto* src = static_cast<const std::uint8_t*>(mapped.pData);
  const int src_stride = static_cast<int>(mapped.RowPitch);

  if (out_w == crop.w && out_h == crop.h) {
    for (int y = 0; y < out_h; ++y) {
      std::memcpy(out_bgra_.data() + static_cast<std::size_t>(y) * out_stride, src + static_cast<std::size_t>(y) * src_stride,
                  static_cast<std::size_t>(out_stride));
    }
  } else {
    scale_bgra_bilinear(src, crop.w, crop.h, src_stride, out_bgra_.data(), out_w, out_h, out_stride);
  }

  unmap();

  if (!sink_->writeFrame(out_bgra_.data(), static_cast<unsigned>(out_stride))) {
    set_error("shm writeFrame failed");
    return;
  }

  last_write_ts_ms_ = t0;
  const std::uint64_t fid = ++frame_id_;
  if (on_frame_) on_frame_(fid, f8::cppsdk::now_ms());
}

void Win32WgcCapture::set_error(std::string err) {
  if (err.empty()) {
    if (on_error_) on_error_(std::string{});
    return;
  }
  spdlog::warn("wgc capture error: {}", err);
  if (on_error_) on_error_(std::move(err));
}

void Win32WgcCapture::scale_bgra_bilinear(const std::uint8_t* src, int src_w, int src_h, int src_stride, std::uint8_t* dst,
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

#else

namespace f8::screencap {

Win32WgcCapture::Win32WgcCapture(std::string, std::shared_ptr<f8::cppsdk::VideoSharedMemorySink>) {}
Win32WgcCapture::~Win32WgcCapture() = default;
void Win32WgcCapture::configure(std::string, double, int, std::string, std::string, std::string) {}
void Win32WgcCapture::restart() {}
void Win32WgcCapture::tick() {}
void Win32WgcCapture::close_capture() {}
bool Win32WgcCapture::open_capture(std::string& err) {
  err = "not supported";
  return false;
}
void Win32WgcCapture::pump_capture() {}
void Win32WgcCapture::set_error(std::string) {}
void Win32WgcCapture::scale_bgra_bilinear(const std::uint8_t*, int, int, int, std::uint8_t*, int, int, int) {}

}  // namespace f8::screencap

#endif
