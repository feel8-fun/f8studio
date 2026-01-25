#pragma once

#include <array>
#include <functional>
#include <string>

#define SDL_MAIN_HANDLED 1
#include <SDL3/SDL.h>

namespace f8::implayer {

class MpvPlayer;

class ImPlayerGui {
 public:
  struct Callbacks {
    std::function<void(const std::string& url)> open;
    std::function<void()> play;
    std::function<void()> pause;
    std::function<void()> stop;
    std::function<void(double position_seconds)> seek;
    std::function<void(double volume01)> set_volume;
  };

  ImPlayerGui();
  ~ImPlayerGui();

  ImPlayerGui(const ImPlayerGui&) = delete;
  ImPlayerGui& operator=(const ImPlayerGui&) = delete;

  bool start(SDL_Window* window, SDL_GLContext gl_context);
  void stop();

  void processEvent(SDL_Event* ev);
  void renderOverlay(const MpvPlayer& player, const Callbacks& cb, const std::string& last_error);

  bool wantsRepaint() const { return dirty_; }
  void clearRepaintFlag() { dirty_ = false; }

 private:
  std::array<char, 1024> url_buf_{};
  float volume01_ = 1.0f;
  bool started_ = false;
  bool dirty_ = true;
};

}  // namespace f8::implayer
