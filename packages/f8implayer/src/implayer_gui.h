#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

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
    std::function<void(bool loop)> set_loop;
    std::function<void(const std::string& hwdec)> set_hwdec;
    std::function<void(int extra_frames)> set_hwdec_extra_frames;
    std::function<void(const std::string& fbo_format)> set_fbo_format;
    std::function<void()> fit_view;
    std::function<void()> toggle_fullscreen;

    std::function<void(int index)> playlist_select;
    std::function<void()> playlist_next;
    std::function<void()> playlist_prev;
  };

  ImPlayerGui();
  ~ImPlayerGui();

  ImPlayerGui(const ImPlayerGui&) = delete;
  ImPlayerGui& operator=(const ImPlayerGui&) = delete;

  bool start(SDL_Window* window, SDL_GLContext gl_context);
  void stop();

  void processEvent(SDL_Event* ev);
  void renderOverlay(const MpvPlayer& player, const Callbacks& cb, const std::string& last_error,
                     const std::vector<std::string>& playlist, int playlist_index, bool playing, bool loop,
                     double tick_fps_ema, double tick_ms_ema);
  bool wantsCaptureKeyboard() const;

  bool wantsRepaint() const { return dirty_; }
  void clearRepaintFlag() { dirty_ = false; }

 private:
  std::array<char, 1024> url_buf_{};
  std::array<char, 32> hwdec_buf_{};
  std::array<char, 16> hwdec_extra_frames_buf_{};
  std::array<char, 32> fbo_format_buf_{};
  float volume01_ = 1.0f;
  float seek_pos_ = 0.0f;
  bool seeking_ = false;
  bool show_playlist_ = false;
  bool show_stats_ = false;
  std::size_t last_playlist_size_ = 0;
  double last_interaction_time_s_ = 0.0;
  bool started_ = false;
  bool dirty_ = true;
};

}  // namespace f8::implayer
