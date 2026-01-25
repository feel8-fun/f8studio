#include "sdl_video_window.h"

#include <algorithm>
#include <cmath>

#include <glad/glad.h>
#include <spdlog/spdlog.h>

#include "gl_loader.h"
#include "mpv_player.h"

namespace f8::implayer {

namespace {

void aspect_fit_rect(unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h, int& x0, int& y0, int& x1, int& y1) {
  x0 = 0;
  y0 = 0;
  x1 = static_cast<int>(dst_w);
  y1 = static_cast<int>(dst_h);
  if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) return;

  const double src_aspect = static_cast<double>(src_w) / static_cast<double>(src_h);
  const double dst_aspect = static_cast<double>(dst_w) / static_cast<double>(dst_h);
  if (dst_aspect > src_aspect) {
    const unsigned fit_w = static_cast<unsigned>(static_cast<double>(dst_h) * src_aspect);
    const int pad = static_cast<int>(dst_w - fit_w) / 2;
    x0 = pad;
    x1 = pad + static_cast<int>(fit_w);
  } else {
    const unsigned fit_h = static_cast<unsigned>(static_cast<double>(dst_w) / src_aspect);
    const int pad = static_cast<int>(dst_h - fit_h) / 2;
    y0 = pad;
    y1 = pad + static_cast<int>(fit_h);
  }
}

}  // namespace

SdlVideoWindow::SdlVideoWindow(Config cfg) : cfg_(std::move(cfg)) {}

SdlVideoWindow::~SdlVideoWindow() { stop(); }

bool SdlVideoWindow::start() {
  if (started_) return true;

  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
    spdlog::error("SDL_Init failed: {}", SDL_GetError());
    return false;
  }

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_WindowFlags flags = SDL_WINDOW_OPENGL;
  if (cfg_.resizable) flags = static_cast<SDL_WindowFlags>(flags | SDL_WINDOW_RESIZABLE);

  window_ = SDL_CreateWindow(cfg_.title.c_str(), cfg_.width, cfg_.height, flags);
  if (!window_) {
    spdlog::error("SDL_CreateWindow failed: {}", SDL_GetError());
    SDL_Quit();
    return false;
  }

  gl_context_ = SDL_GL_CreateContext(window_);
  if (!gl_context_) {
    spdlog::error("SDL_GL_CreateContext failed: {}", SDL_GetError());
    SDL_DestroyWindow(window_);
    window_ = nullptr;
    SDL_Quit();
    return false;
  }

  if (!SDL_GL_MakeCurrent(window_, gl_context_)) {
    spdlog::error("SDL_GL_MakeCurrent failed: {}", SDL_GetError());
    stop();
    return false;
  }

  SDL_GL_SetSwapInterval(cfg_.vsync ? 1 : 0);

  if (!EnsureOpenGLFunctionsLoaded()) {
    spdlog::error("OpenGL function loader failed (glad)");
    stop();
    return false;
  }

  updateDrawableSize();
  clearBlack();
  SDL_GL_SwapWindow(window_);

  started_ = true;
  wants_close_ = false;
  needs_redraw_ = true;
  return true;
}

void SdlVideoWindow::stop() {
  if (!window_ && !gl_context_) {
    if (started_) SDL_Quit();
    started_ = false;
    return;
  }

  if (window_) {
    SDL_GL_MakeCurrent(window_, nullptr);
  }
  if (gl_context_) {
    SDL_GL_DestroyContext(gl_context_);
    gl_context_ = nullptr;
  }
  if (window_) {
    SDL_DestroyWindow(window_);
    window_ = nullptr;
  }
  SDL_Quit();
  started_ = false;
}

bool SdlVideoWindow::pumpEvents(const EventCallback& on_event) {
  if (!started_ || !window_) return false;
  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    if (on_event) on_event(ev);
    switch (ev.type) {
      case SDL_EVENT_QUIT:
        wants_close_ = true;
        break;
      case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        wants_close_ = true;
        break;
      case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
      case SDL_EVENT_WINDOW_RESIZED:
      case SDL_EVENT_WINDOW_EXPOSED:
        needs_redraw_ = true;
        break;
      default:
        break;
    }
  }

  if (wants_close_) return false;
  if (needs_redraw_) updateDrawableSize();
  return true;
}

bool SdlVideoWindow::makeCurrent() {
  if (!started_ || !window_ || !gl_context_) return false;
  return SDL_GL_MakeCurrent(window_, gl_context_);
}

void SdlVideoWindow::present(const MpvPlayer& player, const OverlayCallback& overlay, const ViewTransform& view) {
  if (!started_ || !window_) return;
  if (!makeCurrent()) return;

  updateDrawableSize();
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glViewport(0, 0, static_cast<GLint>(drawable_w_), static_cast<GLint>(drawable_h_));
  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);

  const unsigned src_w = player.videoWidth();
  const unsigned src_h = player.videoHeight();
  const unsigned src_fbo = player.videoFboId();
  if (src_w > 0 && src_h > 0 && src_fbo != 0) {
    int x0, y0, x1, y1;
    aspect_fit_rect(src_w, src_h, drawable_w_, drawable_h_, x0, y0, x1, y1);

    const float zoom = std::clamp(view.zoom, 0.1f, 10.0f);
    const float base_w = static_cast<float>(x1 - x0);
    const float base_h = static_cast<float>(y1 - y0);
    const float cx = static_cast<float>(x0) + base_w * 0.5f;
    const float cy = static_cast<float>(y0) + base_h * 0.5f;
    const float w = base_w * zoom;
    const float h = base_h * zoom;

    const float dx0 = cx - w * 0.5f + view.pan_x;
    const float dy0 = cy - h * 0.5f + view.pan_y;
    const float dx1 = dx0 + w;
    const float dy1 = dy0 + h;

    glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(src_fbo));
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, 0, static_cast<GLint>(src_w), static_cast<GLint>(src_h),
                      static_cast<GLint>(std::lround(dx0)), static_cast<GLint>(std::lround(dy0)),
                      static_cast<GLint>(std::lround(dx1)), static_cast<GLint>(std::lround(dy1)), GL_COLOR_BUFFER_BIT,
                      GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  }

  if (overlay) overlay();
  SDL_GL_SwapWindow(window_);
  needs_redraw_ = false;
}

void SdlVideoWindow::updateDrawableSize() {
  if (!window_) return;
  int w = 0;
  int h = 0;
  if (!SDL_GetWindowSizeInPixels(window_, &w, &h)) {
    w = cfg_.width;
    h = cfg_.height;
  }
  drawable_w_ = static_cast<unsigned>(std::max(0, w));
  drawable_h_ = static_cast<unsigned>(std::max(0, h));
}

void SdlVideoWindow::clearBlack() {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, static_cast<GLint>(drawable_w_), static_cast<GLint>(drawable_h_));
  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
}

}  // namespace f8::implayer
