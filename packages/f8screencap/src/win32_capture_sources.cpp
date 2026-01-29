#include "win32_capture_sources.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <sstream>

#if defined(_WIN32)
#include <Windows.h>
#include <dwmapi.h>
#endif

namespace f8::screencap::win32 {

namespace {

bool parse_int(const std::string& s, int& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  out = static_cast<int>(v);
  return true;
}

bool parse_uintptr_hex(const std::string& s, std::uintptr_t& out) {
  if (s.empty()) return false;
  std::string t = s;
  if (t.rfind("0x", 0) == 0 || t.rfind("0X", 0) == 0) {
    t = t.substr(2);
  }
  if (t.empty()) return false;
  for (unsigned char ch : t) {
    if (!std::isxdigit(ch)) return false;
  }
  char* end = nullptr;
  unsigned long long v = std::strtoull(t.c_str(), &end, 16);
  if (!end || *end != '\0') return false;
  out = static_cast<std::uintptr_t>(v);
  return true;
}

RectI to_recti(long l, long t, long r, long b) {
  RectI out;
  out.x = static_cast<int>(l);
  out.y = static_cast<int>(t);
  out.w = static_cast<int>(r - l);
  out.h = static_cast<int>(b - t);
  return out;
}

#if defined(_WIN32)
std::string wide_to_utf8(const std::wstring& ws) {
  if (ws.empty()) return {};
  const int need = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
  if (need <= 0) return {};
  std::string out(static_cast<std::size_t>(need), '\0');
  WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), out.data(), need, nullptr, nullptr);
  return out;
}
#endif

}  // namespace

RectI virtual_screen_rect() {
  RectI r;
#if defined(_WIN32)
  const int x = GetSystemMetrics(SM_XVIRTUALSCREEN);
  const int y = GetSystemMetrics(SM_YVIRTUALSCREEN);
  const int w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
  const int h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
  r.x = x;
  r.y = y;
  r.w = w;
  r.h = h;
#endif
  return r;
}

#if defined(_WIN32)
struct EnumCtx {
  std::vector<MonitorInfo> monitors;
};

BOOL CALLBACK enum_monitor_proc(HMONITOR mon, HDC, LPRECT, LPARAM lp) {
  auto* ctx = reinterpret_cast<EnumCtx*>(lp);
  if (!ctx) return FALSE;

  MONITORINFOEXW mi;
  std::memset(&mi, 0, sizeof(mi));
  mi.cbSize = sizeof(mi);
  if (!GetMonitorInfoW(mon, &mi)) return TRUE;

  MonitorInfo out;
  out.handle = reinterpret_cast<std::uintptr_t>(mon);
  out.rect = to_recti(mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom);
  out.work_rect = to_recti(mi.rcWork.left, mi.rcWork.top, mi.rcWork.right, mi.rcWork.bottom);
  out.primary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;

  // szDevice looks like: "\\\\.\\DISPLAY1"
  {
    const wchar_t* w = mi.szDevice;
    if (w && *w) {
      out.name = wide_to_utf8(std::wstring(w));
    }
  }

  ctx->monitors.push_back(std::move(out));
  return TRUE;
}
#endif

std::vector<MonitorInfo> enumerate_monitors() {
  std::vector<MonitorInfo> out;
#if defined(_WIN32)
  EnumCtx ctx;
  EnumDisplayMonitors(nullptr, nullptr, &enum_monitor_proc, reinterpret_cast<LPARAM>(&ctx));
  ctx.monitors.shrink_to_fit();

  std::sort(ctx.monitors.begin(), ctx.monitors.end(), [](const MonitorInfo& a, const MonitorInfo& b) {
    if (a.primary != b.primary) return a.primary > b.primary;  // primary first
    if (a.rect.y != b.rect.y) return a.rect.y < b.rect.y;
    return a.rect.x < b.rect.x;
  });

  for (std::size_t i = 0; i < ctx.monitors.size(); ++i) {
    ctx.monitors[i].id = static_cast<int>(i);
  }
  out = std::move(ctx.monitors);
#endif
  return out;
}

bool try_get_monitor_rect(int display_id, RectI& out, std::string& err) {
  err.clear();
  const auto mons = enumerate_monitors();
  if (mons.empty()) {
    err = "no monitors found";
    return false;
  }
  if (display_id < 0 || display_id >= static_cast<int>(mons.size())) {
    err = "displayId out of range";
    return false;
  }
  out = mons[static_cast<std::size_t>(display_id)].rect;
  if (out.w <= 0 || out.h <= 0) {
    err = "invalid monitor rect";
    return false;
  }
  return true;
}

bool try_get_monitor_handle(int display_id, std::uintptr_t& out_hmonitor, std::string& err) {
  err.clear();
  out_hmonitor = 0;
  const auto mons = enumerate_monitors();
  if (mons.empty()) {
    err = "no monitors found";
    return false;
  }
  if (display_id < 0 || display_id >= static_cast<int>(mons.size())) {
    err = "displayId out of range";
    return false;
  }
  out_hmonitor = mons[static_cast<std::size_t>(display_id)].handle;
  if (out_hmonitor == 0) {
    err = "invalid monitor handle";
    return false;
  }
  return true;
}

bool try_parse_csv_rect(const std::string& csv, RectI& out, std::string& err) {
  err.clear();
  std::istringstream ss(csv);
  std::string item;
  std::vector<std::string> parts;
  while (std::getline(ss, item, ',')) {
    while (!item.empty() && std::isspace(static_cast<unsigned char>(item.front()))) item.erase(item.begin());
    while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back()))) item.pop_back();
    if (!item.empty()) parts.push_back(item);
  }
  if (parts.size() != 4) {
    err = "expected x,y,w,h";
    return false;
  }
  int x = 0, y = 0, w = 0, h = 0;
  if (!parse_int(parts[0], x) || !parse_int(parts[1], y) || !parse_int(parts[2], w) || !parse_int(parts[3], h)) {
    err = "invalid integer in rect";
    return false;
  }
  if (w <= 0 || h <= 0) {
    err = "w/h must be > 0";
    return false;
  }
  out = RectI{x, y, w, h};
  return true;
}

bool try_parse_csv_size(const std::string& csv, int& out_w, int& out_h, std::string& err) {
  err.clear();
  std::istringstream ss(csv);
  std::string item;
  std::vector<std::string> parts;
  while (std::getline(ss, item, ',')) {
    while (!item.empty() && std::isspace(static_cast<unsigned char>(item.front()))) item.erase(item.begin());
    while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back()))) item.pop_back();
    if (!item.empty()) parts.push_back(item);
  }
  if (parts.size() != 2) {
    err = "expected w,h";
    return false;
  }
  int w = 0, h = 0;
  if (!parse_int(parts[0], w) || !parse_int(parts[1], h)) {
    err = "invalid integer in size";
    return false;
  }
  if (w < 0 || h < 0) {
    err = "w/h must be >= 0";
    return false;
  }
  out_w = w;
  out_h = h;
  return true;
}

bool try_parse_window_id(const std::string& window_id, std::uintptr_t& out_hwnd, std::string& err) {
  err.clear();
  out_hwnd = 0;
  std::string s = window_id;
  if (s.empty()) {
    err = "windowId is empty";
    return false;
  }
  // Accept "win32:hwnd:0x...."
  const std::string pfx = "win32:hwnd:";
  if (s.rfind(pfx, 0) == 0) {
    s = s.substr(pfx.size());
  }
  std::uintptr_t hwnd = 0;
  if (!parse_uintptr_hex(s, hwnd) || hwnd == 0) {
    err = "invalid windowId (expected hwnd hex)";
    return false;
  }
  out_hwnd = hwnd;
  return true;
}

bool try_get_window_rect(std::uintptr_t hwnd_u, RectI& out, std::string& out_title, std::uint32_t& out_pid,
                         std::string& err) {
  err.clear();
  out_title.clear();
  out_pid = 0;
#if !defined(_WIN32)
  (void)hwnd_u;
  (void)out;
  err = "not supported";
  return false;
#else
  const HWND hwnd = reinterpret_cast<HWND>(hwnd_u);
  if (!IsWindow(hwnd)) {
    err = "hwnd is not a window";
    return false;
  }
  if (IsIconic(hwnd)) {
    err = "window is minimized";
    return false;
  }

  RECT rc{};
  // Prefer DWM extended frame bounds (includes drop shadow).
  HRESULT hr = DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &rc, sizeof(rc));
  if (FAILED(hr)) {
    if (!GetWindowRect(hwnd, &rc)) {
      err = "GetWindowRect failed";
      return false;
    }
  }
  out = to_recti(rc.left, rc.top, rc.right, rc.bottom);
  if (out.w <= 0 || out.h <= 0) {
    err = "invalid window rect";
    return false;
  }

  // Title
  {
    wchar_t buf[512]{};
    const int n = GetWindowTextW(hwnd, buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])));
    if (n > 0) {
      out_title = wide_to_utf8(std::wstring(buf, buf + n));
    }
  }

  DWORD pid = 0;
  (void)GetWindowThreadProcessId(hwnd, &pid);
  out_pid = static_cast<std::uint32_t>(pid);
  return true;
#endif
}

}  // namespace f8::screencap::win32
