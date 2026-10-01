#include "x11_picker.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__) && !defined(_WIN32)
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#if defined(F8_HAVE_XRANDR)
#include <X11/extensions/Xrandr.h>
#endif
#endif

namespace f8::screencap::x11 {

namespace {

#if defined(__linux__) && !defined(_WIN32)
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

unsigned long make_half_brightness_and_mask_pixel(const Visual* vis) {
  if (!vis) return 0;
  const auto r = mask_info(static_cast<std::uint32_t>(vis->red_mask));
  const auto g = mask_info(static_cast<std::uint32_t>(vis->green_mask));
  const auto b = mask_info(static_cast<std::uint32_t>(vis->blue_mask));

  auto half_mask = [](int bits) -> std::uint32_t {
    if (bits <= 0) return 0;
    if (bits >= 31) return 0;
    const std::uint32_t full = (1u << bits) - 1u;
    return full >> 1u;
  };

  std::uint32_t px = 0;
  if (r.bits > 0) px |= (half_mask(r.bits) << r.shift);
  if (g.bits > 0) px |= (half_mask(g.bits) << g.shift);
  if (b.bits > 0) px |= (half_mask(b.bits) << b.shift);
  return static_cast<unsigned long>(px);
}
#endif

struct OverlayContext {
#if defined(__linux__) && !defined(_WIN32)
  Display* dpy = nullptr;
  int screen = 0;
  Window root = 0;
  Window overlay = 0;
  Cursor cursor = 0;
  GC copy_gc = 0;
  GC border_gc = 0;
  GC dim_gc = 0;
  Pixmap bg_px = 0;
  Pixmap stipple = 0;
  int sw = 0;
  int sh = 0;
#endif
};

#if defined(__linux__) && !defined(_WIN32)
void overlay_close(OverlayContext& ctx) {
  if (!ctx.dpy) return;
  if (ctx.copy_gc) XFreeGC(ctx.dpy, ctx.copy_gc);
  if (ctx.border_gc) XFreeGC(ctx.dpy, ctx.border_gc);
  if (ctx.dim_gc) XFreeGC(ctx.dpy, ctx.dim_gc);
  if (ctx.stipple) XFreePixmap(ctx.dpy, ctx.stipple);
  if (ctx.bg_px) XFreePixmap(ctx.dpy, ctx.bg_px);
  XUngrabPointer(ctx.dpy, CurrentTime);
  XUngrabKeyboard(ctx.dpy, CurrentTime);
  if (ctx.overlay) {
    XUnmapWindow(ctx.dpy, ctx.overlay);
    XDestroyWindow(ctx.dpy, ctx.overlay);
  }
  if (ctx.cursor) XFreeCursor(ctx.dpy, ctx.cursor);
  XCloseDisplay(ctx.dpy);
  ctx = {};
}

bool overlay_open(OverlayContext& ctx, std::string& err) {
  err.clear();
  ctx = {};
  ctx.dpy = XOpenDisplay(nullptr);
  if (!ctx.dpy) {
    err = "XOpenDisplay failed (is DISPLAY set?)";
    return false;
  }

  ctx.screen = DefaultScreen(ctx.dpy);
  ctx.root = RootWindow(ctx.dpy, ctx.screen);
  ctx.sw = DisplayWidth(ctx.dpy, ctx.screen);
  ctx.sh = DisplayHeight(ctx.dpy, ctx.screen);
  if (ctx.sw <= 0 || ctx.sh <= 0) {
    err = "invalid screen size";
    overlay_close(ctx);
    return false;
  }

  // Snapshot root for redraw. This avoids XOR trails and works without a compositor.
  XImage* bg_img =
      XGetImage(ctx.dpy, ctx.root, 0, 0, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(ctx.sh), AllPlanes, ZPixmap);
  if (!bg_img || !bg_img->data) {
    if (bg_img) XDestroyImage(bg_img);
    err = "XGetImage failed (cannot snapshot screen)";
    overlay_close(ctx);
    return false;
  }
  if (bg_img->depth != DefaultDepth(ctx.dpy, ctx.screen)) {
    XDestroyImage(bg_img);
    err = "unsupported visual depth for picker overlay";
    overlay_close(ctx);
    return false;
  }

  XSetWindowAttributes attrs{};
  attrs.override_redirect = True;
  attrs.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
  ctx.overlay =
      XCreateWindow(ctx.dpy, ctx.root, 0, 0, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(ctx.sh), 0,
                    DefaultDepth(ctx.dpy, ctx.screen), InputOutput, DefaultVisual(ctx.dpy, ctx.screen),
                    CWOverrideRedirect | CWEventMask, &attrs);
  if (!ctx.overlay) {
    XDestroyImage(bg_img);
    err = "XCreateWindow failed";
    overlay_close(ctx);
    return false;
  }

  ctx.cursor = XCreateFontCursor(ctx.dpy, XC_crosshair);
  if (ctx.cursor) XDefineCursor(ctx.dpy, ctx.overlay, ctx.cursor);

  XMapRaised(ctx.dpy, ctx.overlay);
  XFlush(ctx.dpy);

  const int grab_ptr =
      XGrabPointer(ctx.dpy, ctx.overlay, False, ButtonPressMask | ButtonReleaseMask | PointerMotionMask, GrabModeAsync, GrabModeAsync,
                   None, ctx.cursor, CurrentTime);
  if (grab_ptr != GrabSuccess) {
    XDestroyImage(bg_img);
    err = "XGrabPointer failed";
    overlay_close(ctx);
    return false;
  }
  (void)XGrabKeyboard(ctx.dpy, ctx.overlay, False, GrabModeAsync, GrabModeAsync, CurrentTime);

  ctx.bg_px =
      XCreatePixmap(ctx.dpy, ctx.overlay, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(ctx.sh), static_cast<unsigned>(bg_img->depth));
  if (!ctx.bg_px) {
    XDestroyImage(bg_img);
    err = "XCreatePixmap failed";
    overlay_close(ctx);
    return false;
  }
  ctx.copy_gc = XCreateGC(ctx.dpy, ctx.bg_px, 0, nullptr);
  XPutImage(ctx.dpy, ctx.bg_px, ctx.copy_gc, bg_img, 0, 0, 0, 0, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(ctx.sh));
  XDestroyImage(bg_img);

  XGCValues border_gcv{};
  border_gcv.function = GXcopy;
  border_gcv.line_width = 3;
  border_gcv.foreground = WhitePixel(ctx.dpy, ctx.screen);
  ctx.border_gc = XCreateGC(ctx.dpy, ctx.overlay, GCFunction | GCForeground | GCLineWidth, &border_gcv);

  // Dim GC (best-effort).
  const unsigned long and_mask = make_half_brightness_and_mask_pixel(DefaultVisual(ctx.dpy, ctx.screen));
  if (and_mask != 0) {
    XGCValues dim_gcv{};
    dim_gcv.function = GXand;
    dim_gcv.foreground = and_mask;
    ctx.dim_gc = XCreateGC(ctx.dpy, ctx.overlay, GCFunction | GCForeground, &dim_gcv);
  } else {
    static const char kStippleBits[] = {
        static_cast<char>(0xAA), static_cast<char>(0x55), static_cast<char>(0xAA), static_cast<char>(0x55),
        static_cast<char>(0xAA), static_cast<char>(0x55), static_cast<char>(0xAA), static_cast<char>(0x55),
    };
    ctx.stipple = XCreateBitmapFromData(ctx.dpy, ctx.overlay, kStippleBits, 8, 8);
    XGCValues dim_gcv{};
    dim_gcv.function = GXcopy;
    dim_gcv.foreground = BlackPixel(ctx.dpy, ctx.screen);
    dim_gcv.fill_style = FillStippled;
    dim_gcv.stipple = ctx.stipple;
    ctx.dim_gc = XCreateGC(ctx.dpy, ctx.overlay, GCFunction | GCForeground | GCFillStyle | GCStipple, &dim_gcv);
  }

  return true;
}

void overlay_redraw(const OverlayContext& ctx, const RectI* sel) {
  if (!ctx.dpy || !ctx.overlay) return;
  XCopyArea(ctx.dpy, ctx.bg_px, ctx.overlay, ctx.copy_gc, 0, 0, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(ctx.sh), 0, 0);

  if (!sel || sel->w <= 0 || sel->h <= 0) {
    XFlush(ctx.dpy);
    return;
  }

  int x = sel->x;
  int y = sel->y;
  int w = sel->w;
  int h = sel->h;
  x = std::clamp(x, 0, ctx.sw);
  y = std::clamp(y, 0, ctx.sh);
  w = std::clamp(w, 0, ctx.sw);
  h = std::clamp(h, 0, ctx.sh);
  if (x + w > ctx.sw) w = ctx.sw - x;
  if (y + h > ctx.sh) h = ctx.sh - y;

  const int top_h = y;
  const int bottom_y = y + h;
  const int bottom_h = ctx.sh - bottom_y;
  const int left_w = x;
  const int right_x = x + w;
  const int right_w = ctx.sw - right_x;

  if (top_h > 0) XFillRectangle(ctx.dpy, ctx.overlay, ctx.dim_gc, 0, 0, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(top_h));
  if (bottom_h > 0) {
    XFillRectangle(ctx.dpy, ctx.overlay, ctx.dim_gc, 0, bottom_y, static_cast<unsigned>(ctx.sw), static_cast<unsigned>(bottom_h));
  }
  if (left_w > 0 && h > 0) XFillRectangle(ctx.dpy, ctx.overlay, ctx.dim_gc, 0, y, static_cast<unsigned>(left_w), static_cast<unsigned>(h));
  if (right_w > 0 && h > 0) {
    XFillRectangle(ctx.dpy, ctx.overlay, ctx.dim_gc, right_x, y, static_cast<unsigned>(right_w), static_cast<unsigned>(h));
  }

  XDrawRectangle(ctx.dpy, ctx.overlay, ctx.border_gc, x, y, static_cast<unsigned>(std::max(0, w - 1)),
                 static_cast<unsigned>(std::max(0, h - 1)));
  XFlush(ctx.dpy);
}

bool get_window_rect_title_pid(Display* dpy, Window w, RectI& out, std::string& title, std::uint32_t& pid) {
  out = {};
  title.clear();
  pid = 0;
  if (!dpy || w == 0) return false;

  XWindowAttributes attr{};
  if (!XGetWindowAttributes(dpy, w, &attr)) return false;
  if (attr.map_state != IsViewable) return false;
  if (attr.c_class != InputOutput) return false;
  if (attr.override_redirect) return false;

  int x = 0;
  int y = 0;
  Window child = 0;
  const Window root = RootWindow(dpy, DefaultScreen(dpy));
  if (!XTranslateCoordinates(dpy, w, root, 0, 0, &x, &y, &child)) {
    x = attr.x;
    y = attr.y;
  }
  out = RectI{x, y, attr.width, attr.height};
  if (out.w <= 0 || out.h <= 0) return false;

  Atom utf8 = XInternAtom(dpy, "UTF8_STRING", False);
  Atom net_wm_name = XInternAtom(dpy, "_NET_WM_NAME", False);
  Atom type = 0;
  int format = 0;
  unsigned long nitems = 0;
  unsigned long bytes_after = 0;
  unsigned char* data = nullptr;
  if (Success ==
          XGetWindowProperty(dpy, w, net_wm_name, 0, 1024, False, utf8, &type, &format, &nitems, &bytes_after, &data) &&
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

  Atom net_wm_pid = XInternAtom(dpy, "_NET_WM_PID", False);
  data = nullptr;
  if (Success == XGetWindowProperty(dpy, w, net_wm_pid, 0, 1, False, XA_CARDINAL, &type, &format, &nitems, &bytes_after, &data) &&
      data && nitems >= 1) {
    pid = *reinterpret_cast<std::uint32_t*>(data);
    XFree(data);
  }
  return true;
}

Window topmost_window_at(Display* dpy, Window root, Window ignore, int x, int y) {
  if (!dpy || root == 0) return 0;
  Window root_ret = 0;
  Window parent_ret = 0;
  Window* children = nullptr;
  unsigned int n = 0;
  if (!XQueryTree(dpy, root, &root_ret, &parent_ret, &children, &n)) return 0;

  Window found = 0;
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    const Window w = children[i];
    if (w == 0 || w == ignore) continue;
    RectI rc{};
    std::string title;
    std::uint32_t pid = 0;
    if (!get_window_rect_title_pid(dpy, w, rc, title, pid)) continue;
    if (x >= rc.x && x < rc.x + rc.w && y >= rc.y && y < rc.y + rc.h) {
      found = w;
      break;
    }
  }

  if (children) XFree(children);
  return found;
}

std::vector<DisplayInfo> enumerate_monitors(Display* dpy) {
  std::vector<DisplayInfo> out;
  if (!dpy) return out;
  const int screen = DefaultScreen(dpy);
  const Window root = RootWindow(dpy, screen);

#if defined(F8_HAVE_XRANDR)
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
    return out;
  }
  if (mons) XRRFreeMonitors(mons);
#endif

  DisplayInfo di;
  di.id = 0;
  di.primary = true;
  di.name = "X11 Screen 0";
  di.rect = RectI{0, 0, DisplayWidth(dpy, screen), DisplayHeight(dpy, screen)};
  di.work_rect = di.rect;
  out.push_back(std::move(di));
  return out;
}
#endif

PickRegionResult pick_region_blocking() {
  PickRegionResult r;

#if !defined(__linux__) || defined(_WIN32)
  r.ok = false;
  r.error = "not supported";
  return r;
#else
  OverlayContext ctx;
  std::string err;
  if (!overlay_open(ctx, err)) {
    r.ok = false;
    r.error = err;
    return r;
  }

  bool have_start = false;
  int sx = 0;
  int sy = 0;
  int lx = 0;
  int ly = 0;

  const auto compute = [&](int x0, int y0, int x1, int y1, int& ox, int& oy, int& ow, int& oh) {
    ox = std::min(x0, x1);
    oy = std::min(y0, y1);
    ow = std::max(0, std::abs(x1 - x0));
    oh = std::max(0, std::abs(y1 - y0));
  };

  RectI sel{};
  bool have_sel = false;

  auto redraw = [&]() { overlay_redraw(ctx, have_sel ? &sel : nullptr); };

  bool done = false;
  bool canceled = false;
  while (!done) {
    XEvent ev{};
    XNextEvent(ctx.dpy, &ev);

    if (ev.type == Expose) {
      redraw();
    } else if (ev.type == ButtonPress) {
      if (ev.xbutton.button != Button1) continue;
      have_start = true;
      sx = ev.xbutton.x_root;
      sy = ev.xbutton.y_root;
      lx = sx;
      ly = sy;
      compute(sx, sy, lx, ly, sel.x, sel.y, sel.w, sel.h);
      have_sel = true;
      redraw();
    } else if (ev.type == MotionNotify) {
      if (!have_start) continue;
      // Coalesce motion events.
      while (XCheckTypedEvent(ctx.dpy, MotionNotify, &ev)) {
      }
      lx = ev.xmotion.x_root;
      ly = ev.xmotion.y_root;
      compute(sx, sy, lx, ly, sel.x, sel.y, sel.w, sel.h);
      have_sel = true;
      redraw();
    } else if (ev.type == ButtonRelease) {
      if (!have_start || ev.xbutton.button != Button1) continue;
      lx = ev.xbutton.x_root;
      ly = ev.xbutton.y_root;
      int ox = 0, oy = 0, ow = 0, oh = 0;
      compute(sx, sy, lx, ly, ox, oy, ow, oh);
      if (ow <= 0 || oh <= 0) {
        canceled = true;
      } else {
        r.ok = true;
        r.rect = RectI{ox, oy, ow, oh};
      }
      done = true;
    } else if (ev.type == KeyPress) {
      const KeySym ks = XLookupKeysym(&ev.xkey, 0);
      if (ks == XK_Escape) {
        canceled = true;
        done = true;
      }
    }
  }

  overlay_close(ctx);

  if (!r.ok) {
    r.error = canceled ? "canceled" : "failed";
  }
  return r;
#endif
}

}  // namespace

PickDisplayResult pick_display_blocking() {
  PickDisplayResult r;
#if !defined(__linux__) || defined(_WIN32)
  r.ok = false;
  r.error = "not supported";
  return r;
#else
  OverlayContext ctx;
  std::string err;
  if (!overlay_open(ctx, err)) {
    r.ok = false;
    r.error = err;
    return r;
  }

  const auto displays = enumerate_monitors(ctx.dpy);
  if (displays.empty()) {
    overlay_close(ctx);
    r.ok = false;
    r.error = "no displays";
    return r;
  }

  RectI sel{};
  bool have_sel = false;
  int sel_id = 0;
  auto redraw = [&]() { overlay_redraw(ctx, have_sel ? &sel : nullptr); };

  bool done = false;
  bool canceled = false;
  while (!done) {
    XEvent ev{};
    XNextEvent(ctx.dpy, &ev);
    if (ev.type == Expose) {
      redraw();
      continue;
    }
    if (ev.type == MotionNotify) {
      while (XCheckTypedEvent(ctx.dpy, MotionNotify, &ev)) {
      }
      const int x = ev.xmotion.x_root;
      const int y = ev.xmotion.y_root;
      have_sel = false;
      for (const auto& d : displays) {
        if (x >= d.rect.x && x < d.rect.x + d.rect.w && y >= d.rect.y && y < d.rect.y + d.rect.h) {
          sel = d.rect;
          sel_id = d.id;
          have_sel = true;
          break;
        }
      }
      redraw();
      continue;
    }
    if (ev.type == ButtonRelease) {
      if (ev.xbutton.button != Button1) continue;
      if (!have_sel) {
        canceled = true;
      } else {
        const auto* chosen = static_cast<const DisplayInfo*>(nullptr);
        for (const auto& d : displays) {
          if (d.id == sel_id) {
            chosen = &d;
            break;
          }
        }
        if (chosen) {
          r.ok = true;
          r.display_id = chosen->id;
          r.rect = chosen->rect;
          r.name = chosen->name;
          r.primary = chosen->primary;
        } else {
          canceled = true;
        }
      }
      done = true;
      continue;
    }
    if (ev.type == KeyPress) {
      const KeySym ks = XLookupKeysym(&ev.xkey, 0);
      if (ks == XK_Escape) {
        canceled = true;
        done = true;
      }
    }
  }

  overlay_close(ctx);
  if (!r.ok) r.error = canceled ? "canceled" : "failed";
  return r;
#endif
}

PickWindowResult pick_window_blocking() {
  PickWindowResult r;
#if !defined(__linux__) || defined(_WIN32)
  r.ok = false;
  r.error = "not supported";
  return r;
#else
  OverlayContext ctx;
  std::string err;
  if (!overlay_open(ctx, err)) {
    r.ok = false;
    r.error = err;
    return r;
  }

  RectI sel{};
  bool have_sel = false;
  Window sel_win = 0;
  std::string sel_title;
  std::uint32_t sel_pid = 0;
  auto redraw = [&]() { overlay_redraw(ctx, have_sel ? &sel : nullptr); };

  bool done = false;
  bool canceled = false;
  while (!done) {
    XEvent ev{};
    XNextEvent(ctx.dpy, &ev);
    if (ev.type == Expose) {
      redraw();
      continue;
    }
    if (ev.type == MotionNotify) {
      while (XCheckTypedEvent(ctx.dpy, MotionNotify, &ev)) {
      }
      const int x = ev.xmotion.x_root;
      const int y = ev.xmotion.y_root;
      const Window w = topmost_window_at(ctx.dpy, ctx.root, ctx.overlay, x, y);
      if (w != sel_win) {
        sel_win = w;
        sel_title.clear();
        sel_pid = 0;
        have_sel = false;
        if (sel_win) {
          if (get_window_rect_title_pid(ctx.dpy, sel_win, sel, sel_title, sel_pid)) {
            have_sel = true;
          }
        }
        redraw();
      }
      continue;
    }
    if (ev.type == ButtonRelease) {
      if (ev.xbutton.button != Button1) continue;
      if (!have_sel || !sel_win) {
        canceled = true;
      } else {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "x11:win:0x%llx", static_cast<unsigned long long>(sel_win));
        r.ok = true;
        r.window_id = buf;
        r.rect = sel;
        r.title = sel_title;
        r.pid = sel_pid;
      }
      done = true;
      continue;
    }
    if (ev.type == KeyPress) {
      const KeySym ks = XLookupKeysym(&ev.xkey, 0);
      if (ks == XK_Escape) {
        canceled = true;
        done = true;
      }
    }
  }

  overlay_close(ctx);
  if (!r.ok) r.error = canceled ? "canceled" : "failed";
  return r;
#endif
}

void X11Picker::pick_region_async(PickRegionCallback cb) {
  std::thread([cb = std::move(cb)]() mutable {
    PickRegionResult r = pick_region_blocking();
    if (cb) cb(std::move(r));
  }).detach();
}

void X11Picker::pick_display_async(PickDisplayCallback cb) {
  std::thread([cb = std::move(cb)]() mutable {
    PickDisplayResult r = pick_display_blocking();
    if (cb) cb(std::move(r));
  }).detach();
}

void X11Picker::pick_window_async(PickWindowCallback cb) {
  std::thread([cb = std::move(cb)]() mutable {
    PickWindowResult r = pick_window_blocking();
    if (cb) cb(std::move(r));
  }).detach();
}

}  // namespace f8::screencap::x11
