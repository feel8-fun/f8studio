#include "sdl_video_window.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <glad/glad.h>
#include <spdlog/spdlog.h>

#include "gl_loader.h"
#include "mpv_player.h"

namespace f8::implayer {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr const char* kVrVertexShader = R"(
#version 330 core
out vec2 v_ndc;
void main() {
  vec2 pos;
  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
  else if (gl_VertexID == 1) pos = vec2(3.0, -1.0);
  else pos = vec2(-1.0, 3.0);
  v_ndc = pos;
  gl_Position = vec4(pos, 0.0, 1.0);
}
)";

constexpr const char* kVrFragmentShader = R"(
#version 330 core
in vec2 v_ndc;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform float uAspect;
uniform float uYawRad;
uniform float uPitchRad;
uniform float uTanHalfFov;
uniform int uMode;     // 1=mono, 2=sbs
uniform int uSbsEye;   // 0=left, 1=right

void main() {
  vec3 dir = normalize(vec3(v_ndc.x * uTanHalfFov * uAspect, -v_ndc.y * uTanHalfFov, 1.0));

  float cp = cos(uPitchRad);
  float sp = sin(uPitchRad);
  float cy = cos(uYawRad);
  float sy = sin(uYawRad);

  vec3 d1 = vec3(dir.x, cp * dir.y - sp * dir.z, sp * dir.y + cp * dir.z);
  vec3 d = vec3(cy * d1.x + sy * d1.z, d1.y, -sy * d1.x + cy * d1.z);

  float u = atan(d.x, d.z) / (2.0 * 3.14159265358979323846) + 0.5;
  float v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / 3.14159265358979323846;

  if (uMode == 2) {
    if (uSbsEye == 0) u *= 0.5;
    else u = 0.5 + u * 0.5;
  }

  FragColor = texture(uTexture, vec2(fract(u), clamp(v, 0.0, 1.0)));
}
)";

void aspect_fit_rect(unsigned src_w, unsigned src_h, unsigned dst_w, unsigned dst_h, int& x0, int& y0, int& x1,
                     int& y1) {
  x0 = 0;
  y0 = 0;
  x1 = static_cast<int>(dst_w);
  y1 = static_cast<int>(dst_h);
  if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0)
    return;

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

SdlVideoWindow::~SdlVideoWindow() {
  stop();
}

bool SdlVideoWindow::start() {
  if (started_)
    return true;

  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
    spdlog::error("SDL_Init failed: {}", SDL_GetError());
    return false;
  }

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_WindowFlags flags = SDL_WINDOW_OPENGL;
  if (cfg_.resizable)
    flags = static_cast<SDL_WindowFlags>(flags | SDL_WINDOW_RESIZABLE);

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
    if (started_)
      SDL_Quit();
    started_ = false;
    return;
  }

  if (window_ && gl_context_) {
    SDL_GL_MakeCurrent(window_, gl_context_);
  }
  destroyVrRenderer();
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
  if (!started_ || !window_)
    return false;
  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    if (on_event)
      on_event(ev);
    processEvent(ev);
  }

  if (wants_close_)
    return false;
  if (needs_redraw_)
    updateDrawableSize();
  return true;
}

void SdlVideoWindow::processEvent(const SDL_Event& ev) {
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

  if (needs_redraw_)
    updateDrawableSize();
}

bool SdlVideoWindow::makeCurrent() {
  if (!started_ || !window_ || !gl_context_)
    return false;
  return SDL_GL_MakeCurrent(window_, gl_context_);
}

void SdlVideoWindow::present(const MpvPlayer& player, const OverlayCallback& overlay) {
  present(player, overlay, ViewTransform{}, VrViewState{});
}

void SdlVideoWindow::present(const MpvPlayer& player, const OverlayCallback& overlay, const ViewTransform& view) {
  present(player, overlay, view, VrViewState{});
}

void SdlVideoWindow::present(const MpvPlayer& player, const OverlayCallback& overlay, const ViewTransform& view,
                             const VrViewState& vr) {
  if (!started_ || !window_)
    return;
  if (!makeCurrent())
    return;

  updateDrawableSize();
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glViewport(0, 0, static_cast<GLint>(drawable_w_), static_cast<GLint>(drawable_h_));
  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);

  const unsigned src_w = player.videoWidth();
  const unsigned src_h = player.videoHeight();
  const unsigned src_fbo = player.videoFboId();
  const unsigned src_texture = player.videoTextureId();
  if (src_w > 0 && src_h > 0 && src_fbo != 0) {
    const bool want_vr = vr.mode != ProjectionMode::Flat2D;
    bool rendered_vr = false;
    if (want_vr && src_texture != 0) {
      rendered_vr = renderVrFrame(src_texture, vr);
    }

    if (!rendered_vr) {
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
  }

  if (overlay)
    overlay();
  SDL_GL_SwapWindow(window_);
  needs_redraw_ = false;
}

bool SdlVideoWindow::ensureVrRenderer() {
  if (vr_program_ != 0 && vr_vao_ != 0) {
    return true;
  }

  const unsigned vs = compileShader(GL_VERTEX_SHADER, kVrVertexShader);
  if (vs == 0) {
    return false;
  }
  const unsigned fs = compileShader(GL_FRAGMENT_SHADER, kVrFragmentShader);
  if (fs == 0) {
    glDeleteShader(vs);
    return false;
  }

  const unsigned program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glDeleteShader(vs);
  glDeleteShader(fs);

  GLint linked = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    logProgramFailure(program);
    glDeleteProgram(program);
    return false;
  }

  unsigned vao = 0;
  glGenVertexArrays(1, &vao);
  if (vao == 0) {
    glDeleteProgram(program);
    spdlog::error("VR renderer init failed: glGenVertexArrays returned 0");
    return false;
  }

  vr_program_ = program;
  vr_vao_ = vao;
  vr_u_texture_ = glGetUniformLocation(vr_program_, "uTexture");
  vr_u_aspect_ = glGetUniformLocation(vr_program_, "uAspect");
  vr_u_yaw_rad_ = glGetUniformLocation(vr_program_, "uYawRad");
  vr_u_pitch_rad_ = glGetUniformLocation(vr_program_, "uPitchRad");
  vr_u_tan_half_fov_ = glGetUniformLocation(vr_program_, "uTanHalfFov");
  vr_u_mode_ = glGetUniformLocation(vr_program_, "uMode");
  vr_u_sbs_eye_ = glGetUniformLocation(vr_program_, "uSbsEye");
  return true;
}

void SdlVideoWindow::destroyVrRenderer() {
  if (vr_vao_ != 0) {
    glDeleteVertexArrays(1, &vr_vao_);
    vr_vao_ = 0;
  }
  if (vr_program_ != 0) {
    glDeleteProgram(vr_program_);
    vr_program_ = 0;
  }
  vr_u_texture_ = -1;
  vr_u_aspect_ = -1;
  vr_u_yaw_rad_ = -1;
  vr_u_pitch_rad_ = -1;
  vr_u_tan_half_fov_ = -1;
  vr_u_mode_ = -1;
  vr_u_sbs_eye_ = -1;
  vr_renderer_failed_once_ = false;
}

bool SdlVideoWindow::renderVrFrame(unsigned texture_id, const VrViewState& vr) {
  if (!ensureVrRenderer()) {
    if (!vr_renderer_failed_once_) {
      vr_renderer_failed_once_ = true;
      spdlog::warn("VR renderer unavailable; falling back to 2D blit");
    }
    return false;
  }
  vr_renderer_failed_once_ = false;
  if (drawable_h_ == 0) {
    return false;
  }

  const float fov_deg = std::clamp(vr.fov_deg, 50.0f, 120.0f);
  const float yaw_rad = vr.yaw_deg * (kPi / 180.0f);
  const float pitch_rad = vr.pitch_deg * (kPi / 180.0f);
  const float half_fov_rad = 0.5f * fov_deg * (kPi / 180.0f);
  const float tan_half_fov = std::tan(half_fov_rad);
  const float aspect = static_cast<float>(drawable_w_) / static_cast<float>(drawable_h_);

  int mode = 0;
  if (vr.mode == ProjectionMode::EquirectMono) {
    mode = 1;
  } else if (vr.mode == ProjectionMode::EquirectSbs) {
    mode = 2;
  }
  const int sbs_eye = (vr.sbs_eye == 0) ? 0 : 1;

  glUseProgram(vr_program_);
  glBindVertexArray(vr_vao_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_id);

  if (vr_u_texture_ >= 0)
    glUniform1i(vr_u_texture_, 0);
  if (vr_u_aspect_ >= 0)
    glUniform1f(vr_u_aspect_, aspect);
  if (vr_u_yaw_rad_ >= 0)
    glUniform1f(vr_u_yaw_rad_, yaw_rad);
  if (vr_u_pitch_rad_ >= 0)
    glUniform1f(vr_u_pitch_rad_, pitch_rad);
  if (vr_u_tan_half_fov_ >= 0)
    glUniform1f(vr_u_tan_half_fov_, tan_half_fov);
  if (vr_u_mode_ >= 0)
    glUniform1i(vr_u_mode_, mode);
  if (vr_u_sbs_eye_ >= 0)
    glUniform1i(vr_u_sbs_eye_, sbs_eye);

  glDisable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDrawArrays(GL_TRIANGLES, 0, 3);

  glBindTexture(GL_TEXTURE_2D, 0);
  glBindVertexArray(0);
  glUseProgram(0);
  return true;
}

unsigned SdlVideoWindow::compileShader(unsigned type, const char* source) {
  const unsigned shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (ok != GL_TRUE) {
    logShaderFailure(shader, type == GL_VERTEX_SHADER ? "vertex" : "fragment");
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

void SdlVideoWindow::logShaderFailure(unsigned shader, const char* stage) {
  GLint len = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  std::string info;
  if (len > 0) {
    info.resize(static_cast<std::size_t>(len));
    glGetShaderInfoLog(shader, len, nullptr, info.data());
  }
  spdlog::error("VR {} shader compile failed: {}", stage ? stage : "unknown", info);
}

void SdlVideoWindow::logProgramFailure(unsigned program) {
  GLint len = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
  std::string info;
  if (len > 0) {
    info.resize(static_cast<std::size_t>(len));
    glGetProgramInfoLog(program, len, nullptr, info.data());
  }
  spdlog::error("VR shader program link failed: {}", info);
}

void SdlVideoWindow::updateDrawableSize() {
  if (!window_)
    return;
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

bool SdlVideoWindow::isFullscreen() const {
  if (!window_)
    return false;
  const SDL_WindowFlags flags = SDL_GetWindowFlags(window_);
  return (flags & SDL_WINDOW_FULLSCREEN) != 0;
}

bool SdlVideoWindow::setFullscreen(bool fullscreen) {
  if (!window_)
    return false;
  if (!SDL_SetWindowFullscreen(window_, fullscreen)) {
    spdlog::warn("SDL_SetWindowFullscreen failed: {}", SDL_GetError());
    return false;
  }
  (void)SDL_SyncWindow(window_);
  needs_redraw_ = true;
  updateDrawableSize();
  return true;
}

bool SdlVideoWindow::toggleFullscreen() {
  return setFullscreen(!isFullscreen());
}

}  // namespace f8::implayer
