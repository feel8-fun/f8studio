#include "x11_capture_sources.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

#if defined(__linux__) && !defined(_WIN32)
#if defined(F8_HAVE_XRANDR)
#include <X11/extensions/Xrandr.h>
#endif
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace f8::screencap::x11 {

namespace {

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}

bool parse_ints_csv(const std::string& csv, std::vector<int>& out, std::string& err) {
  err.clear();
  out.clear();
  std::string s = trim(csv);
  if (s.empty()) {
    err = "empty csv";
    return false;
  }
  std::replace(s.begin(), s.end(), ',', ' ');
  std::istringstream ss(s);
  int v = 0;
  while (ss >> v) out.push_back(v);
  if (out.empty()) {
    err = "failed to parse csv";
    return false;
  }
  return true;
}

}  // namespace

bool try_parse_csv_rect(const std::string& csv, RectI& out, std::string& err) {
  std::vector<int> v;
  if (!parse_ints_csv(csv, v, err)) return false;
  if (v.size() < 4) {
    err = "expected x,y,w,h";
    return false;
  }
  out.x = v[0];
  out.y = v[1];
  out.w = v[2];
  out.h = v[3];
  if (out.w <= 0 || out.h <= 0) {
    err = "w/h must be > 0";
    return false;
  }
  return true;
}

bool try_parse_csv_size(const std::string& csv, int& out_w, int& out_h, std::string& err) {
  err.clear();
  out_w = 0;
  out_h = 0;
  const std::string s = trim(csv);
  if (s.empty()) return true;
  std::vector<int> v;
  if (!parse_ints_csv(s, v, err)) return false;
  if (v.size() < 2) {
    err = "expected w,h";
    return false;
  }
  out_w = v[0];
  out_h = v[1];
  if (out_w < 0 || out_h < 0) {
    err = "w/h must be >= 0";
    return false;
  }
  return true;
}

bool try_parse_window_id(const std::string& window_id, std::uint64_t& out_xid, std::string& err) {
  err.clear();
  out_xid = 0;
  std::string s = trim(window_id);
  if (s.empty()) {
    err = "window id empty";
    return false;
  }
  // Accept: "x11:win:0x1234", "x11:0x1234", "0x1234", "1234"
  if (s.rfind("x11:", 0) == 0) s = s.substr(4);
  if (s.rfind("win:", 0) == 0) s = s.substr(4);
  s = trim(s);
  if (s.rfind("0x", 0) != 0 && s.rfind("0X", 0) != 0) {
    // If purely digits, interpret as hex if it looks like an XID, else decimal.
    bool all_digits = true;
    for (char c : s) {
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        all_digits = false;
        break;
      }
    }
    if (!all_digits) {
      err = "invalid window id";
      return false;
    }
  }

  try {
    std::size_t idx = 0;
    const int base = (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) ? 16 : 10;
    const std::uint64_t v = std::stoull(s, &idx, base);
    if (idx != s.size() || v == 0) {
      err = "invalid window id";
      return false;
    }
    out_xid = v;
    return true;
  } catch (...) {
    err = "invalid window id";
    return false;
  }
}

std::vector<DisplayInfo> enumerate_displays(std::string& err) {
  err.clear();
  std::vector<DisplayInfo> out;

#if defined(__linux__) && !defined(_WIN32)
  Display* dpy = XOpenDisplay(nullptr);
  if (!dpy) {
    err = "XOpenDisplay failed (is DISPLAY set?)";
    return out;
  }

#if defined(F8_HAVE_XRANDR)
  // Prefer XRandR monitors when available (multi-monitor under a single X11 screen).
  const int screen = DefaultScreen(dpy);
  const Window root = RootWindow(dpy, screen);
  int nmon = 0;
  XRRMonitorInfo* mons = XRRGetMonitors(dpy, root, True, &nmon);
  if (mons && nmon > 0) {
    out.reserve(static_cast<std::size_t>(nmon));
    for (int i = 0; i < nmon; ++i) {
      DisplayInfo di;
      di.id = i;
      di.primary = mons[i].primary != 0;
      di.rect = RectI{mons[i].x, mons[i].y, mons[i].width, mons[i].height};
      di.work_rect = di.rect;
      std::string name = "Monitor " + std::to_string(i);
      if (mons[i].name != None) {
        char* atom_name = XGetAtomName(dpy, mons[i].name);
        if (atom_name) {
          name = atom_name;
          XFree(atom_name);
        }
      }
      di.name = std::move(name);
      out.push_back(std::move(di));
    }
    XRRFreeMonitors(mons);
    XCloseDisplay(dpy);
    return out;
  }
  if (mons) XRRFreeMonitors(mons);
#endif

  // Fallback: X11 screens.
  const int screens = ScreenCount(dpy);
  for (int i = 0; i < screens; ++i) {
    DisplayInfo di;
    di.id = i;
    di.primary = (i == DefaultScreen(dpy));
    char namebuf[64];
    std::snprintf(namebuf, sizeof(namebuf), "X11 Screen %d", i);
    di.name = namebuf;
    di.rect = RectI{0, 0, DisplayWidth(dpy, i), DisplayHeight(dpy, i)};
    di.work_rect = di.rect;
    out.push_back(std::move(di));
  }
  XCloseDisplay(dpy);
  return out;
#else
  err = "not supported";
  return out;
#endif
}

bool try_get_window_rect(std::uint64_t xid, RectI& out, std::string& title, std::uint32_t& pid, std::string& err) {
  err.clear();
  out = {};
  title.clear();
  pid = 0;

#if defined(__linux__) && !defined(_WIN32)
  Display* dpy = XOpenDisplay(nullptr);
  if (!dpy) {
    err = "XOpenDisplay failed";
    return false;
  }

  const Window w = static_cast<Window>(xid);
  XWindowAttributes attr{};
  if (!XGetWindowAttributes(dpy, w, &attr)) {
    XCloseDisplay(dpy);
    err = "XGetWindowAttributes failed";
    return false;
  }

  int x = 0;
  int y = 0;
  Window child = 0;
  const Window root = RootWindow(dpy, DefaultScreen(dpy));
  if (!XTranslateCoordinates(dpy, w, root, 0, 0, &x, &y, &child)) {
    x = attr.x;
    y = attr.y;
  }

  out = RectI{x, y, attr.width, attr.height};

  // Title (best-effort): _NET_WM_NAME (UTF8) then fallback to XFetchName.
  Atom utf8 = XInternAtom(dpy, "UTF8_STRING", False);
  Atom net_wm_name = XInternAtom(dpy, "_NET_WM_NAME", False);
  Atom type = 0;
  int format = 0;
  unsigned long nitems = 0;
  unsigned long bytes_after = 0;
  unsigned char* data = nullptr;
  if (Success == XGetWindowProperty(dpy, w, net_wm_name, 0, 1024, False, utf8, &type, &format, &nitems, &bytes_after, &data) &&
      data) {
    title.assign(reinterpret_cast<const char*>(data), reinterpret_cast<const char*>(data) + nitems);
    XFree(data);
  } else {
    char* nm = nullptr;
    if (XFetchName(dpy, w, &nm) && nm) {
      title = nm;
      XFree(nm);
    }
  }

  // PID (best-effort): _NET_WM_PID
  Atom net_wm_pid = XInternAtom(dpy, "_NET_WM_PID", False);
  data = nullptr;
  if (Success == XGetWindowProperty(dpy, w, net_wm_pid, 0, 1, False, XA_CARDINAL, &type, &format, &nitems, &bytes_after, &data) &&
      data && nitems >= 1) {
    pid = *reinterpret_cast<std::uint32_t*>(data);
    XFree(data);
  }

  XCloseDisplay(dpy);
  return (out.w > 0 && out.h > 0);
#else
  (void)xid;
  err = "not supported";
  return false;
#endif
}

}  // namespace f8::screencap::x11
