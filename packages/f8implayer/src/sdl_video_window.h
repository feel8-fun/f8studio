#pragma once

#include <string>
#include <functional>

#define SDL_MAIN_HANDLED 1
#include <SDL3/SDL.h>

namespace f8::implayer {

class MpvPlayer;

class SdlVideoWindow {
 public:
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

  bool makeCurrent();

  // Present the latest mpv frame (blits the player's framebuffer into the window).
  // If no mpv frame exists yet, clears to black.
  using OverlayCallback = std::function<void()>;
  void present(const MpvPlayer& player, const OverlayCallback& overlay = {});

  bool wantsClose() const { return wants_close_; }
  bool needsRedraw() const { return needs_redraw_; }
  void clearRedrawFlag() { needs_redraw_ = false; }

  unsigned drawableWidth() const { return drawable_w_; }
  unsigned drawableHeight() const { return drawable_h_; }

  SDL_Window* sdlWindow() const { return window_; }
  SDL_GLContext glContext() const { return gl_context_; }

 private:
  void updateDrawableSize();
  void clearBlack();

  Config cfg_;
  SDL_Window* window_ = nullptr;
  SDL_GLContext gl_context_ = nullptr;

  bool started_ = false;
  bool wants_close_ = false;
  bool needs_redraw_ = true;
  unsigned drawable_w_ = 0;
  unsigned drawable_h_ = 0;
};

}  // namespace f8::implayer
