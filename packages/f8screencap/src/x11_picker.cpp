#include "x11_picker.h"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <utility>

#if defined(__linux__) && !defined(_WIN32)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
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

PickRegionResult pick_region_blocking() {
  PickRegionResult r;

#if !defined(__linux__) || defined(_WIN32)
  r.ok = false;
  r.error = "not supported";
  return r;
#else
  Display* dpy = XOpenDisplay(nullptr);
  if (!dpy) {
    r.ok = false;
    r.error = "XOpenDisplay failed (is DISPLAY set?)";
    return r;
  }

  const int screen = DefaultScreen(dpy);
  const Window root = RootWindow(dpy, screen);
  const int sw = DisplayWidth(dpy, screen);
  const int sh = DisplayHeight(dpy, screen);
  if (sw <= 0 || sh <= 0) {
    XCloseDisplay(dpy);
    r.ok = false;
    r.error = "invalid screen size";
    return r;
  }

  // Take a snapshot of the root window so we can draw an opaque overlay (works without compositor).
  XImage* bg_img = XGetImage(dpy, root, 0, 0, static_cast<unsigned>(sw), static_cast<unsigned>(sh), AllPlanes, ZPixmap);
  if (!bg_img || !bg_img->data) {
    if (bg_img) XDestroyImage(bg_img);
    XCloseDisplay(dpy);
    r.ok = false;
    r.error = "XGetImage failed (cannot snapshot screen)";
    return r;
  }
  if (bg_img->depth != DefaultDepth(dpy, screen)) {
    XDestroyImage(bg_img);
    XCloseDisplay(dpy);
    r.ok = false;
    r.error = "unsupported visual depth for picker overlay";
    return r;
  }

  XSetWindowAttributes attrs{};
  attrs.override_redirect = True;
  attrs.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
  Window overlay =
      XCreateWindow(dpy, root, 0, 0, static_cast<unsigned>(sw), static_cast<unsigned>(sh), 0, DefaultDepth(dpy, screen), InputOutput,
                    DefaultVisual(dpy, screen), CWOverrideRedirect | CWEventMask, &attrs);
  if (!overlay) {
    XDestroyImage(bg_img);
    XCloseDisplay(dpy);
    r.ok = false;
    r.error = "XCreateWindow failed";
    return r;
  }

  Cursor cursor = XCreateFontCursor(dpy, XC_crosshair);
  if (cursor) XDefineCursor(dpy, overlay, cursor);

  XMapRaised(dpy, overlay);
  XFlush(dpy);

  const int grab_ptr =
      XGrabPointer(dpy, overlay, False, ButtonPressMask | ButtonReleaseMask | PointerMotionMask, GrabModeAsync, GrabModeAsync, None,
                   cursor, CurrentTime);
  if (grab_ptr != GrabSuccess) {
    XUnmapWindow(dpy, overlay);
    XDestroyWindow(dpy, overlay);
    if (cursor) XFreeCursor(dpy, cursor);
    XDestroyImage(bg_img);
    XCloseDisplay(dpy);
    r.ok = false;
    r.error = "XGrabPointer failed";
    return r;
  }
  (void)XGrabKeyboard(dpy, overlay, False, GrabModeAsync, GrabModeAsync, CurrentTime);

  Pixmap bg_px = XCreatePixmap(dpy, overlay, static_cast<unsigned>(sw), static_cast<unsigned>(sh), static_cast<unsigned>(bg_img->depth));
  if (!bg_px) {
    XUngrabPointer(dpy, CurrentTime);
    XUngrabKeyboard(dpy, CurrentTime);
    XUnmapWindow(dpy, overlay);
    XDestroyWindow(dpy, overlay);
    if (cursor) XFreeCursor(dpy, cursor);
    XDestroyImage(bg_img);
    XCloseDisplay(dpy);
    r.ok = false;
    r.error = "XCreatePixmap failed";
    return r;
  }

  GC copy_gc = XCreateGC(dpy, bg_px, 0, nullptr);
  XPutImage(dpy, bg_px, copy_gc, bg_img, 0, 0, 0, 0, static_cast<unsigned>(sw), static_cast<unsigned>(sh));
  XDestroyImage(bg_img);
  bg_img = nullptr;

  // Border GC (high contrast).
  XGCValues border_gcv{};
  border_gcv.function = GXcopy;
  border_gcv.line_width = 2;
  border_gcv.foreground = WhitePixel(dpy, screen);
  GC border_gc = XCreateGC(dpy, overlay, GCFunction | GCForeground | GCLineWidth, &border_gcv);

  // Dimming GC: try a portable "half brightness" AND mask; fallback to stipple if it looks unsupported.
  const unsigned long and_mask = make_half_brightness_and_mask_pixel(DefaultVisual(dpy, screen));
  GC dim_gc = nullptr;
  Pixmap stipple = 0;
  if (and_mask != 0) {
    XGCValues dim_gcv{};
    dim_gcv.function = GXand;
    dim_gcv.foreground = and_mask;
    dim_gc = XCreateGC(dpy, overlay, GCFunction | GCForeground, &dim_gcv);
  } else {
    static const char kStippleBits[] = {
        static_cast<char>(0xAA), static_cast<char>(0x55), static_cast<char>(0xAA), static_cast<char>(0x55),
        static_cast<char>(0xAA), static_cast<char>(0x55), static_cast<char>(0xAA), static_cast<char>(0x55),
    };
    stipple = XCreateBitmapFromData(dpy, overlay, kStippleBits, 8, 8);
    XGCValues dim_gcv{};
    dim_gcv.function = GXcopy;
    dim_gcv.foreground = BlackPixel(dpy, screen);
    dim_gcv.fill_style = FillStippled;
    dim_gcv.stipple = stipple;
    dim_gc = XCreateGC(dpy, overlay, GCFunction | GCForeground | GCFillStyle | GCStipple, &dim_gcv);
  }

  auto cleanup = [&]() {
    if (copy_gc) XFreeGC(dpy, copy_gc);
    if (border_gc) XFreeGC(dpy, border_gc);
    if (dim_gc) XFreeGC(dpy, dim_gc);
    if (stipple) XFreePixmap(dpy, stipple);
    if (bg_px) XFreePixmap(dpy, bg_px);
    XUngrabPointer(dpy, CurrentTime);
    XUngrabKeyboard(dpy, CurrentTime);
    XUnmapWindow(dpy, overlay);
    XDestroyWindow(dpy, overlay);
    if (cursor) XFreeCursor(dpy, cursor);
    XCloseDisplay(dpy);
  };

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

  int sel_x = 0, sel_y = 0, sel_w = 0, sel_h = 0;
  bool have_sel = false;

  auto redraw = [&]() {
    XCopyArea(dpy, bg_px, overlay, copy_gc, 0, 0, static_cast<unsigned>(sw), static_cast<unsigned>(sh), 0, 0);

    if (!have_sel || sel_w <= 0 || sel_h <= 0) {
      XFlush(dpy);
      return;
    }

    auto clamp_dim = [](int& v, int lo, int hi) { v = std::clamp(v, lo, hi); };
    int x = sel_x;
    int y = sel_y;
    int w = sel_w;
    int h = sel_h;
    clamp_dim(x, 0, sw);
    clamp_dim(y, 0, sh);
    clamp_dim(w, 0, sw);
    clamp_dim(h, 0, sh);
    if (x + w > sw) w = sw - x;
    if (y + h > sh) h = sh - y;

    const int top_h = y;
    const int bottom_y = y + h;
    const int bottom_h = sh - bottom_y;
    const int left_w = x;
    const int right_x = x + w;
    const int right_w = sw - right_x;

    if (top_h > 0) XFillRectangle(dpy, overlay, dim_gc, 0, 0, static_cast<unsigned>(sw), static_cast<unsigned>(top_h));
    if (bottom_h > 0) XFillRectangle(dpy, overlay, dim_gc, 0, bottom_y, static_cast<unsigned>(sw), static_cast<unsigned>(bottom_h));
    if (left_w > 0 && h > 0) XFillRectangle(dpy, overlay, dim_gc, 0, y, static_cast<unsigned>(left_w), static_cast<unsigned>(h));
    if (right_w > 0 && h > 0) {
      XFillRectangle(dpy, overlay, dim_gc, right_x, y, static_cast<unsigned>(right_w), static_cast<unsigned>(h));
    }

    // Border.
    XDrawRectangle(dpy, overlay, border_gc, x, y, static_cast<unsigned>(std::max(0, w - 1)),
                   static_cast<unsigned>(std::max(0, h - 1)));
    XFlush(dpy);
  };

  bool done = false;
  bool canceled = false;
  while (!done) {
    XEvent ev{};
    XNextEvent(dpy, &ev);

    if (ev.type == Expose) {
      redraw();
    } else if (ev.type == ButtonPress) {
      if (ev.xbutton.button != Button1) continue;
      have_start = true;
      sx = ev.xbutton.x_root;
      sy = ev.xbutton.y_root;
      lx = sx;
      ly = sy;
      compute(sx, sy, lx, ly, sel_x, sel_y, sel_w, sel_h);
      have_sel = true;
      redraw();
    } else if (ev.type == MotionNotify) {
      if (!have_start) continue;
      // Coalesce motion events.
      while (XCheckTypedEvent(dpy, MotionNotify, &ev)) {
      }
      lx = ev.xmotion.x_root;
      ly = ev.xmotion.y_root;
      compute(sx, sy, lx, ly, sel_x, sel_y, sel_w, sel_h);
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

  cleanup();

  if (!r.ok) {
    r.error = canceled ? "canceled" : "failed";
  }
  return r;
#endif
}

}  // namespace

void X11Picker::pick_region_async(PickRegionCallback cb) {
  std::thread([cb = std::move(cb)]() mutable {
    PickRegionResult r = pick_region_blocking();
    if (cb) cb(std::move(r));
  }).detach();
}

}  // namespace f8::screencap::x11
