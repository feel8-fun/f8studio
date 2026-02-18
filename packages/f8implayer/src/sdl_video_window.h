#pragma once

#include <functional>
#include <string>

#include <SDL3/SDL.h>

namespace f8::implayer {

class MpvPlayer;

class SdlVideoWindow {
 public:
  enum class ProjectionMode {
    Flat2D = 0,
    EquirectMono = 1,
    EquirectSbs = 2,
  };

  struct Config {
    std::string title = "f8implayer";
    int width = 1280;
    int height = 720;
    bool resizable = true;
    bool vsync = true;
  };

  explicit SdlVideoWindow(Config cfg);
  ~SdlVideoWindow();

  SdlVideoWindow(const SdlVideoWindow&) = delete;
  SdlVideoWindow& operator=(const SdlVideoWindow&) = delete;

  bool start();
  void stop();

  // Pump OS/window events. Returns false when the window is closing.
  using EventCallback = std::function<void(const SDL_Event&)>;
  bool pumpEvents(const EventCallback& on_event = {});

  // When using SDL_MAIN_USE_CALLBACKS, feed events here from SDL_AppEvent.
  void processEvent(const SDL_Event& ev);

  bool makeCurrent();

  // Present the latest mpv frame (blits the player's framebuffer into the window).
  // If no mpv frame exists yet, clears to black.
  struct ViewTransform {
    float zoom = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
  };

  struct VrViewState {
    ProjectionMode mode = ProjectionMode::Flat2D;
    float yaw_deg = 0.0f;
    float pitch_deg = 0.0f;
    float fov_deg = 90.0f;
    int sbs_eye = 0;
  };

  using OverlayCallback = std::function<void()>;
  void present(const MpvPlayer& player, const OverlayCallback& overlay = {});
  void present(const MpvPlayer& player, const OverlayCallback& overlay, const ViewTransform& view);
  void present(const MpvPlayer& player, const OverlayCallback& overlay, const ViewTransform& view, const VrViewState& vr);

  bool wantsClose() const { return wants_close_; }
  bool needsRedraw() const { return needs_redraw_; }
  void clearRedrawFlag() { needs_redraw_ = false; }

  unsigned drawableWidth() const { return drawable_w_; }
  unsigned drawableHeight() const { return drawable_h_; }

  bool isFullscreen() const;
  bool setFullscreen(bool fullscreen);
  bool toggleFullscreen();

  SDL_Window* sdlWindow() const { return window_; }
  SDL_GLContext glContext() const { return gl_context_; }

 private:
  void updateDrawableSize();
  void clearBlack();
  bool ensureVrRenderer();
  void destroyVrRenderer();
  bool renderVrFrame(unsigned texture_id, const VrViewState& vr);
  static unsigned compileShader(unsigned type, const char* source);
  static void logShaderFailure(unsigned shader, const char* stage);
  static void logProgramFailure(unsigned program);

  Config cfg_;
  SDL_Window* window_ = nullptr;
  SDL_GLContext gl_context_ = nullptr;

  bool started_ = false;
  bool wants_close_ = false;
  bool needs_redraw_ = true;
  unsigned drawable_w_ = 0;
  unsigned drawable_h_ = 0;

  unsigned vr_program_ = 0;
  unsigned vr_vao_ = 0;
  int vr_u_texture_ = -1;
  int vr_u_aspect_ = -1;
  int vr_u_yaw_rad_ = -1;
  int vr_u_pitch_rad_ = -1;
  int vr_u_tan_half_fov_ = -1;
  int vr_u_mode_ = -1;
  int vr_u_sbs_eye_ = -1;
  bool vr_renderer_failed_once_ = false;
};

}  // namespace f8::implayer
