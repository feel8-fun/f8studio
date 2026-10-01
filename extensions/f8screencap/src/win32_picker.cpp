#include "win32_picker.h"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>

#if defined(_WIN32)
#include <Windows.h>
#include <Windowsx.h>
#endif

namespace f8::screencap::win32 {

#if defined(_WIN32)
namespace {

struct PickerThreadState final {
  enum class Kind { kDisplay, kWindow, kRegion };
  Kind kind{};

  bool done = false;
  bool ok = false;

  // display
  int display_id = -1;

  // window
  std::uintptr_t hwnd = 0;
  RectI window_rect{};
  std::string window_title;
  std::uint32_t window_pid = 0;

  // region
  RectI region{};
};

const wchar_t* kPickerWndClass = L"F8ScreenCapPickerOverlay";

// 0xAARRGGBB (little-endian in memory => BB GG RR AA => BGRA as expected for 32bpp DIB)
constexpr std::uint32_t kDimPixel = 0xB0000000u;     // A=0xB0, black
constexpr std::uint32_t kHighlightPixel = 0x10000000u;  // A=0x10, black (nearly un-dim)
constexpr std::uint32_t kBorderPixel = 0xFFFFCC00u;  // A=0xFF, bright yellow

struct OverlayContext final {
  PickerThreadState* st = nullptr;
  RectI virt{};

  HWND hwnd = nullptr;
  HDC mem_dc = nullptr;
  HBITMAP dib = nullptr;
  void* dib_bits = nullptr;
  int w = 0;
  int h = 0;

  bool dragging = false;
  POINT drag_start{};

  RectI highlight{};
  bool highlight_valid = false;

  std::vector<std::uint32_t> pixels;
};

thread_local OverlayContext* g_ctx = nullptr;
thread_local HHOOK g_kbd_hook = nullptr;

LRESULT CALLBACK low_level_keyboard_proc(int nCode, WPARAM wParam, LPARAM lParam) {
  if (nCode == HC_ACTION && g_ctx && g_ctx->st) {
    const auto* ks = reinterpret_cast<const KBDLLHOOKSTRUCT*>(lParam);
    if (ks && (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN)) {
      if (ks->vkCode == VK_ESCAPE) {
        g_ctx->st->ok = false;
        g_ctx->st->done = true;
        PostQuitMessage(0);
        return 1;
      }
    }
  }
  return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

RectI recti_from_screen_rect(const RECT& rc) {
  return RectI{static_cast<int>(rc.left), static_cast<int>(rc.top), static_cast<int>(rc.right - rc.left),
               static_cast<int>(rc.bottom - rc.top)};
}

bool rect_valid(const RectI& r) { return r.w > 0 && r.h > 0; }

RectI clamp_to_virt(const RectI& r, const RectI& virt) {
  RectI out = r;
  const int vx0 = virt.x;
  const int vy0 = virt.y;
  const int vx1 = virt.x + virt.w;
  const int vy1 = virt.y + virt.h;
  int x0 = std::max(out.x, vx0);
  int y0 = std::max(out.y, vy0);
  int x1 = std::min(out.x + out.w, vx1);
  int y1 = std::min(out.y + out.h, vy1);
  out.x = x0;
  out.y = y0;
  out.w = std::max(0, x1 - x0);
  out.h = std::max(0, y1 - y0);
  return out;
}

RectI rect_to_overlay(const RectI& screen, const RectI& virt) {
  return RectI{screen.x - virt.x, screen.y - virt.y, screen.w, screen.h};
}

void draw_border(std::vector<std::uint32_t>& px, int w, int h, const RectI& r, int thickness) {
  if (!rect_valid(r) || w <= 0 || h <= 0) return;
  const int x0 = std::max(0, r.x);
  const int y0 = std::max(0, r.y);
  const int x1 = std::min(w, r.x + r.w);
  const int y1 = std::min(h, r.y + r.h);
  if (x1 <= x0 || y1 <= y0) return;

  const int t = std::max(1, thickness);
  auto setp = [&](int x, int y, std::uint32_t v) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = v;
  };

  for (int dy = 0; dy < t; ++dy) {
    const int yt = y0 + dy;
    const int yb = (y1 - 1) - dy;
    for (int x = x0; x < x1; ++x) {
      setp(x, yt, kBorderPixel);
      setp(x, yb, kBorderPixel);
    }
  }
  for (int dx = 0; dx < t; ++dx) {
    const int xl = x0 + dx;
    const int xr = (x1 - 1) - dx;
    for (int y = y0; y < y1; ++y) {
      setp(xl, y, kBorderPixel);
      setp(xr, y, kBorderPixel);
    }
  }
}

void render_overlay(OverlayContext& ctx) {
  if (!ctx.dib_bits || ctx.w <= 0 || ctx.h <= 0) return;
  const std::size_t count = static_cast<std::size_t>(ctx.w) * static_cast<std::size_t>(ctx.h);
  if (ctx.pixels.size() != count) ctx.pixels.assign(count, kDimPixel);

  std::fill(ctx.pixels.begin(), ctx.pixels.end(), kDimPixel);

  if (ctx.highlight_valid && rect_valid(ctx.highlight)) {
    const RectI r = clamp_to_virt(ctx.highlight, RectI{0, 0, ctx.w, ctx.h});
    if (rect_valid(r)) {
      for (int y = r.y; y < r.y + r.h; ++y) {
        auto* row = ctx.pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ctx.w);
        for (int x = r.x; x < r.x + r.w; ++x) {
          // IMPORTANT: keep alpha non-zero, otherwise mouse clicks can pass through the layered window
          // and our overlay won't receive confirm/cancel clicks.
          row[x] = kHighlightPixel;
        }
      }
      draw_border(ctx.pixels, ctx.w, ctx.h, r, 2);
    }
  }

  std::memcpy(ctx.dib_bits, ctx.pixels.data(), ctx.pixels.size() * sizeof(std::uint32_t));

  SIZE size{ctx.w, ctx.h};
  POINT src_pt{0, 0};
  POINT win_pt{ctx.virt.x, ctx.virt.y};
  BLENDFUNCTION bf{};
  bf.BlendOp = AC_SRC_OVER;
  bf.SourceConstantAlpha = 255;
  bf.AlphaFormat = AC_SRC_ALPHA;
  HDC screen = GetDC(nullptr);
  UpdateLayeredWindow(ctx.hwnd, screen, &win_pt, &size, ctx.mem_dc, &src_pt, 0, &bf, ULW_ALPHA);
  ReleaseDC(nullptr, screen);
}

int monitor_id_from_point(const POINT& pt) {
  const auto mons = enumerate_monitors();
  if (mons.empty()) return -1;
  HMONITOR mon = MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
  if (!mon) return -1;
  MONITORINFOEXW mi{};
  mi.cbSize = sizeof(mi);
  if (!GetMonitorInfoW(mon, &mi)) return -1;
  const RectI rc = RectI{mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top};
  for (const auto& m : mons) {
    if (m.rect.x == rc.x && m.rect.y == rc.y && m.rect.w == rc.w && m.rect.h == rc.h) return m.id;
  }
  return 0;
}

HWND hwnd_topmost_at_point_excluding(const POINT& pt, HWND exclude) {
  // Walk top-level windows in z-order (topmost first).
  for (HWND w = GetTopWindow(nullptr); w != nullptr; w = GetWindow(w, GW_HWNDNEXT)) {
    if (!IsWindow(w) || !IsWindowVisible(w)) continue;
    if (w == exclude) continue;
    if (IsIconic(w)) continue;
    HWND root = GetAncestor(w, GA_ROOT);
    if (!root) continue;
    if (root == exclude) continue;
    RECT rc{};
    if (!GetWindowRect(root, &rc)) continue;
    if (PtInRect(&rc, pt)) return root;
  }
  return nullptr;
}

LRESULT CALLBACK overlay_wndproc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  auto* ctx = reinterpret_cast<OverlayContext*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

  switch (msg) {
    case WM_CREATE: {
      auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
      auto* c = cs ? reinterpret_cast<OverlayContext*>(cs->lpCreateParams) : nullptr;
      SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(c));
      return 0;
    }
    case WM_SETCURSOR: {
      SetCursor(LoadCursorW(nullptr, MAKEINTRESOURCEW(32515)));  // IDC_CROSS
      return TRUE;
    }
    case WM_KEYDOWN: {
      if (!ctx || !ctx->st) return 0;
      if (wParam == VK_ESCAPE) {
        ctx->st->ok = false;
        ctx->st->done = true;
        PostQuitMessage(0);
        return 0;
      }
      return 0;
    }
    case WM_RBUTTONDOWN: {
      if (!ctx || !ctx->st) return 0;
      ctx->st->ok = false;
      ctx->st->done = true;
      PostQuitMessage(0);
      return 0;
    }
    case WM_LBUTTONDOWN: {
      if (!ctx || !ctx->st) return 0;
      SetCapture(hwnd);
      if (ctx->st->kind == PickerThreadState::Kind::kRegion) {
        ctx->dragging = true;
        ctx->drag_start.x = GET_X_LPARAM(lParam);
        ctx->drag_start.y = GET_Y_LPARAM(lParam);
        ctx->highlight_valid = false;
        render_overlay(*ctx);
      } else {
        // Confirm current selection.
        const bool ok = (ctx->st->kind == PickerThreadState::Kind::kDisplay) ? (ctx->st->display_id >= 0)
                                                                             : (ctx->st->hwnd != 0);
        ctx->st->ok = ok;
        ctx->st->done = true;
        PostQuitMessage(0);
      }
      return 0;
    }
    case WM_LBUTTONUP: {
      if (!ctx || !ctx->st) return 0;
      if (ctx->st->kind == PickerThreadState::Kind::kRegion && ctx->dragging) {
        ctx->dragging = false;
        ReleaseCapture();

        const int x0 = ctx->drag_start.x;
        const int y0 = ctx->drag_start.y;
        const int x1 = GET_X_LPARAM(lParam);
        const int y1 = GET_Y_LPARAM(lParam);
        const int rx = (x0 < x1) ? x0 : x1;
        const int ry = (y0 < y1) ? y0 : y1;
        const int rw = std::abs(x1 - x0);
        const int rh = std::abs(y1 - y0);

        if (rw <= 1 || rh <= 1) {
          ctx->st->ok = false;
          ctx->st->done = true;
          PostQuitMessage(0);
          return 0;
        }

        ctx->st->region = RectI{ctx->virt.x + rx, ctx->virt.y + ry, rw, rh};
        ctx->st->ok = true;
        ctx->st->done = true;
        PostQuitMessage(0);
      }
      return 0;
    }
    case WM_MOUSEMOVE: {
      if (!ctx || !ctx->st) return 0;
      const POINT local{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
      POINT screen_pt{ctx->virt.x + local.x, ctx->virt.y + local.y};

      if (ctx->st->kind == PickerThreadState::Kind::kDisplay) {
        const int id = monitor_id_from_point(screen_pt);
        if (id >= 0) {
          RectI rc{};
          std::string err;
          if (try_get_monitor_rect(id, rc, err)) {
            ctx->st->display_id = id;
            const RectI ov = rect_to_overlay(rc, ctx->virt);
            ctx->highlight = ov;
            ctx->highlight_valid = true;
            render_overlay(*ctx);
          }
        }
      } else if (ctx->st->kind == PickerThreadState::Kind::kWindow) {
        HWND w = hwnd_topmost_at_point_excluding(screen_pt, hwnd);
        if (w) {
          RectI rc{};
          std::string title;
          std::uint32_t pid = 0;
          std::string err;
          if (try_get_window_rect(reinterpret_cast<std::uintptr_t>(w), rc, title, pid, err)) {
            ctx->st->hwnd = reinterpret_cast<std::uintptr_t>(w);
            ctx->st->window_rect = rc;
            ctx->st->window_title = title;
            ctx->st->window_pid = pid;
            ctx->highlight = rect_to_overlay(rc, ctx->virt);
            ctx->highlight_valid = true;
            render_overlay(*ctx);
          }
        }
      } else if (ctx->st->kind == PickerThreadState::Kind::kRegion) {
        if (ctx->dragging) {
          const int x0 = ctx->drag_start.x;
          const int y0 = ctx->drag_start.y;
          const int x1 = local.x;
          const int y1 = local.y;
          const int rx = (x0 < x1) ? x0 : x1;
          const int ry = (y0 < y1) ? y0 : y1;
          const int rw = std::abs(x1 - x0);
          const int rh = std::abs(y1 - y0);
          ctx->highlight = RectI{rx, ry, rw, rh};
          ctx->highlight_valid = rect_valid(ctx->highlight);
          render_overlay(*ctx);
        }
      }
      return 0;
    }
    case WM_TIMER: {
      // Keep highlight updated even if windows move while hovering.
      if (!ctx || !ctx->st) return 0;
      if (ctx->st->kind != PickerThreadState::Kind::kWindow) return 0;
      if (ctx->st->hwnd == 0) return 0;
      RectI rc{};
      std::string title;
      std::uint32_t pid = 0;
      std::string err;
      if (try_get_window_rect(ctx->st->hwnd, rc, title, pid, err)) {
        ctx->st->window_rect = rc;
        ctx->st->window_title = title;
        ctx->st->window_pid = pid;
        ctx->highlight = rect_to_overlay(rc, ctx->virt);
        ctx->highlight_valid = true;
        render_overlay(*ctx);
      }
      return 0;
    }
    default:
      return DefWindowProcW(hwnd, msg, wParam, lParam);
  }
}

bool ensure_overlay_class() {
  static std::atomic<bool> registered{false};
  if (registered.load(std::memory_order_acquire)) return true;
  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = &overlay_wndproc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.lpszClassName = kPickerWndClass;
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32515));  // IDC_CROSS
  if (!RegisterClassExW(&wc)) {
    const DWORD e = GetLastError();
    if (e != ERROR_CLASS_ALREADY_EXISTS) return false;
  }
  registered.store(true, std::memory_order_release);
  return true;
}

template <class Fn>
void run_picker_thread(PickerThreadState::Kind kind, Fn on_done) {
  std::thread([kind, on_done = std::move(on_done)]() mutable {
    PickerThreadState st;
    st.kind = kind;

    OverlayContext ctx;
    ctx.st = &st;
    g_ctx = &ctx;
    ctx.virt = virtual_screen_rect();
    ctx.w = std::max(1, ctx.virt.w);
    ctx.h = std::max(1, ctx.virt.h);

    if (!ensure_overlay_class()) {
      st.ok = false;
      st.done = true;
      on_done(std::move(st));
      return;
    }

    const DWORD ex = WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW;
    const DWORD style = WS_POPUP;
    ctx.hwnd = CreateWindowExW(ex, kPickerWndClass, L"", style, ctx.virt.x, ctx.virt.y, ctx.w, ctx.h, nullptr, nullptr,
                              GetModuleHandleW(nullptr), &ctx);
    if (!ctx.hwnd) {
      st.ok = false;
      st.done = true;
      on_done(std::move(st));
      return;
    }

    // Create 32bpp DIB + mem DC for UpdateLayeredWindow.
    HDC screen = GetDC(nullptr);
    ctx.mem_dc = CreateCompatibleDC(screen);
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = ctx.w;
    bmi.bmiHeader.biHeight = -ctx.h;  // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    ctx.dib = CreateDIBSection(screen, &bmi, DIB_RGB_COLORS, &ctx.dib_bits, nullptr, 0);
    ReleaseDC(nullptr, screen);
    if (!ctx.mem_dc || !ctx.dib || !ctx.dib_bits) {
      st.ok = false;
      st.done = true;
      if (ctx.dib) DeleteObject(ctx.dib);
      if (ctx.mem_dc) DeleteDC(ctx.mem_dc);
      DestroyWindow(ctx.hwnd);
      on_done(std::move(st));
      return;
    }
    SelectObject(ctx.mem_dc, ctx.dib);

    ShowWindow(ctx.hwnd, SW_SHOW);
    SetForegroundWindow(ctx.hwnd);
    SetFocus(ctx.hwnd);
    SetCapture(ctx.hwnd);
    SetTimer(ctx.hwnd, 1, 33, nullptr);

    g_kbd_hook = SetWindowsHookExW(WH_KEYBOARD_LL, &low_level_keyboard_proc, nullptr, 0);

    // Initial render.
    ctx.highlight_valid = false;
    render_overlay(ctx);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
      if (st.done) break;
    }

    if (g_kbd_hook) {
      UnhookWindowsHookEx(g_kbd_hook);
      g_kbd_hook = nullptr;
    }

    if (ctx.hwnd) {
      KillTimer(ctx.hwnd, 1);
      ReleaseCapture();
      DestroyWindow(ctx.hwnd);
      ctx.hwnd = nullptr;
    }
    if (ctx.dib) {
      DeleteObject(ctx.dib);
      ctx.dib = nullptr;
    }
    if (ctx.mem_dc) {
      DeleteDC(ctx.mem_dc);
      ctx.mem_dc = nullptr;
    }

    g_ctx = nullptr;
    on_done(std::move(st));
  }).detach();
}

}  // namespace
#endif  // _WIN32

void Win32Picker::pick_display_async(OnPickDisplay cb) {
#if defined(_WIN32)
  run_picker_thread(PickerThreadState::Kind::kDisplay,
                    [cb = std::move(cb)](PickerThreadState st) mutable {
                      PickDisplayResult out;
                      out.ok = st.ok;
                      out.display_id = st.display_id;
                      if (!out.ok) out.err = "cancelled";
                      cb(std::move(out));
                    });
#else
  PickDisplayResult out;
  out.ok = false;
  out.err = "not supported";
  cb(std::move(out));
#endif
}

void Win32Picker::pick_window_async(OnPickWindow cb) {
#if defined(_WIN32)
  run_picker_thread(PickerThreadState::Kind::kWindow,
                    [cb = std::move(cb)](PickerThreadState st) mutable {
                      PickWindowResult out;
                      out.ok = st.ok;
                      if (!out.ok) {
                        out.err = "cancelled";
                        cb(std::move(out));
                        return;
                      }
                      out.rect = st.window_rect;
                      out.title = st.window_title;
                      out.pid = st.window_pid;
                      {
                        char buf[64]{};
                        std::snprintf(buf, sizeof(buf), "win32:hwnd:0x%llx",
                                      static_cast<unsigned long long>(st.hwnd));
                        out.window_id = buf;
                      }
                      cb(std::move(out));
                    });
#else
  PickWindowResult out;
  out.ok = false;
  out.err = "not supported";
  cb(std::move(out));
#endif
}

void Win32Picker::pick_region_async(OnPickRegion cb) {
#if defined(_WIN32)
  run_picker_thread(PickerThreadState::Kind::kRegion,
                    [cb = std::move(cb)](PickerThreadState st) mutable {
                      PickRegionResult out;
                      out.ok = st.ok;
                      out.rect = st.region;
                      if (!out.ok) out.err = "cancelled";
                      cb(std::move(out));
                    });
#else
  PickRegionResult out;
  out.ok = false;
  out.err = "not supported";
  cb(std::move(out));
#endif
}

}  // namespace f8::screencap::win32
