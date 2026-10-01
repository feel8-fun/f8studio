#pragma once

#include <string>

#include <SDL3/SDL.h>

#include "sdl_video_window.h"

namespace f8::implayer {

class OpenXrPresenter final {
 public:
  struct Events {
    bool play_pause_pressed = false;
    bool playlist_next_pressed = false;
    bool playlist_prev_pressed = false;
    bool cycle_projection_pressed = false;

    bool seek_absolute_valid = false;
    double seek_absolute_fraction01 = 0.0;  // [0,1]

    double seek_delta_seconds = 0.0;  // relative seek this frame
  };

  struct FrameParams {
    unsigned src_texture = 0;
    unsigned src_width = 0;
    unsigned src_height = 0;

    SdlVideoWindow::ProjectionMode mode = SdlVideoWindow::ProjectionMode::Flat2D;
    int sbs_eye = 0;

    bool playing = false;
    double position_seconds = 0.0;
    double duration_seconds = 0.0;

    // Extra offsets applied on top of the headset pose (useful for mouse-drag).
    float yaw_offset_deg = 0.0f;
    float pitch_offset_deg = 0.0f;
  };

  OpenXrPresenter();
  ~OpenXrPresenter();

  OpenXrPresenter(const OpenXrPresenter&) = delete;
  OpenXrPresenter& operator=(const OpenXrPresenter&) = delete;

  bool start(SDL_Window* sdl_window, SDL_GLContext gl_context, std::string& err);
  void stop();

  // Returns false on fatal error (e.g. runtime lost or explicit exit requested).
  bool renderFrame(const FrameParams& frame, Events* events, std::string& err);

  bool started() const;
  bool exitRequested() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace f8::implayer
