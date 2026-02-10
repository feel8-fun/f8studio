#include "implayer_gui.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <imgui.h>

#include "imgui_backends/imgui_impl_opengl3.h"
#include "imgui_backends/imgui_impl_sdl3.h"
#include "mpv_player.h"

namespace f8::implayer {

namespace {

std::string format_time(double seconds) {
  if (!(seconds >= 0.0))
    seconds = 0.0;
  const auto s = static_cast<long long>(seconds + 0.5);
  const auto hh = s / 3600;
  const auto mm = (s / 60) % 60;
  const auto ss = s % 60;
  char buf[64] = {};
  if (hh > 0) {
    std::snprintf(buf, sizeof(buf), "%lld:%02lld:%02lld", hh, mm, ss);
  } else {
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld", mm, ss);
  }
  return std::string(buf);
}

double bytes_to_mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

ImPlayerGui::ImPlayerGui() {
  url_buf_.fill(0);
  hwdec_buf_.fill(0);
  hwdec_extra_frames_buf_.fill(0);
  fbo_format_buf_.fill(0);
}

ImPlayerGui::~ImPlayerGui() {
  stop();
}

bool ImPlayerGui::start(SDL_Window* window, SDL_GLContext gl_context) {
  if (started_)
    return true;
  if (!window || !gl_context)
    return false;

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();

  if (!ImGui_ImplSDL3_InitForOpenGL(window, gl_context)) {
    stop();
    return false;
  }
  if (!ImGui_ImplOpenGL3_Init("#version 330")) {
    stop();
    return false;
  }

  started_ = true;
  dirty_ = true;
  return true;
}

void ImPlayerGui::stop() {
  if (!started_)
    return;
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  started_ = false;
}

void ImPlayerGui::processEvent(SDL_Event* ev) {
  if (!started_ || !ev)
    return;
  ImGui_ImplSDL3_ProcessEvent(ev);
  dirty_ = true;
}

void ImPlayerGui::renderOverlay(const MpvPlayer& player, const Callbacks& cb, const std::string& last_error,
                                const std::vector<std::string>& playlist, int playlist_index, bool playing, bool loop) {
  if (!started_)
    return;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();

  ImGuiIO& io = ImGui::GetIO();
  const double now_s = ImGui::GetTime();

  const auto* vp = ImGui::GetMainViewport();
  const ImVec2 vpos = vp ? vp->Pos : ImVec2(0, 0);
  const ImVec2 vsize = vp ? vp->Size : ImVec2(1280, 720);

  const bool mouse_in_controls_region = (io.MousePos.y >= (vpos.y + vsize.y - 160.0f));
  if ((io.MouseDelta.x != 0.0f) || (io.MouseDelta.y != 0.0f) || (io.MouseWheel != 0.0f) || (io.MouseWheelH != 0.0f) ||
      ImGui::IsAnyItemActive() || mouse_in_controls_region) {
    last_interaction_time_s_ = now_s;
  }

  const double dur = std::max(0.0, player.durationSeconds());
  const double pos = std::clamp(player.positionSeconds(), 0.0, dur > 0.0 ? dur : player.positionSeconds());

  if (!seeking_) {
    seek_pos_ = static_cast<float>(pos);
  }

  const bool show_controls =
      seeking_ || mouse_in_controls_region || (!playing) || ((now_s - last_interaction_time_s_) <= 1.25);

  if (show_controls) {
    const float w = std::min(820.0f, vsize.x - 24.0f);
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::SetNextWindowPos(ImVec2(vpos.x + (vsize.x - w) * 0.5f, vpos.y + vsize.y - 120.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(w, 0), ImGuiCond_Always);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_AlwaysAutoResize;
    if (ImGui::Begin("##implayer_controls", nullptr, flags)) {
      if (ImGui::Button(playing ? "Pause" : "Play")) {
        if (playing) {
          if (cb.pause)
            cb.pause();
        } else {
          if (cb.play)
            cb.play();
        }
        dirty_ = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Stop")) {
        if (cb.stop)
          cb.stop();
        dirty_ = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Prev")) {
        if (cb.playlist_prev)
          cb.playlist_prev();
        dirty_ = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Next")) {
        if (cb.playlist_next)
          cb.playlist_next();
        dirty_ = true;
      }
      ImGui::SameLine();
      show_playlist_ = show_playlist_ || (playlist.size() > 1);
      if (ImGui::SmallButton(show_playlist_ ? "List:On" : "List:Off")) {
        show_playlist_ = !show_playlist_;
        dirty_ = true;
      }

      ImGui::SameLine();
      if (ImGui::SmallButton(loop ? "Loop:On" : "Loop:Off")) {
        if (cb.set_loop)
          cb.set_loop(!loop);
        dirty_ = true;
      }

      ImGui::SameLine();
      if (ImGui::SmallButton(show_stats_ ? "Stats:On" : "Stats:Off")) {
        show_stats_ = !show_stats_;
        dirty_ = true;
      }

      ImGui::SameLine();
      ImGui::TextDisabled("%s / %s", format_time(pos).c_str(), (dur > 0.0 ? format_time(dur).c_str() : "--:--"));

      const float seek_max = dur > 0.0 ? static_cast<float>(dur) : 0.0f;
      ImGui::SetNextItemWidth(-1);
      const bool seek_enabled = seek_max > 0.001f;
      if (!seek_enabled) {
        ImGui::BeginDisabled();
      }
      if (ImGui::SliderFloat("##seek", &seek_pos_, 0.0f, seek_max, "", ImGuiSliderFlags_AlwaysClamp)) {
        dirty_ = true;
      }
      if (ImGui::IsItemActive())
        seeking_ = true;
      if (ImGui::IsItemDeactivatedAfterEdit()) {
        seeking_ = false;
        if (cb.seek) {
          cb.seek(static_cast<double>(seek_pos_));
          dirty_ = true;
        }
      }
      if (!seek_enabled) {
        ImGui::EndDisabled();
      }

      ImGui::SetNextItemWidth(150);
      if (ImGui::SliderFloat("##volume", &volume01_, 0.0f, 1.0f, "vol=%.2f")) {
        if (cb.set_volume)
          cb.set_volume(static_cast<double>(volume01_));
        dirty_ = true;
      }

      ImGui::SameLine();
      if (ImGui::Button("Open")) {
        ImGui::OpenPopup("##open_popup");
        dirty_ = true;
      }
      if (ImGui::BeginPopup("##open_popup")) {
        ImGui::TextUnformatted("URL / filepath:");
        ImGui::SetNextItemWidth(520);
        if (ImGui::InputText("##url", url_buf_.data(), url_buf_.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
          if (cb.open)
            cb.open(std::string(url_buf_.data()));
          ImGui::CloseCurrentPopup();
          dirty_ = true;
        }
        if (ImGui::Button("Open now")) {
          if (cb.open)
            cb.open(std::string(url_buf_.data()));
          ImGui::CloseCurrentPopup();
          dirty_ = true;
        }
        ImGui::EndPopup();
      }

      if (!last_error.empty()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f), "Error: %s", last_error.c_str());
      }
    }
    ImGui::End();
  }

  if (show_stats_) {
    ImGui::SetNextWindowBgAlpha(0.80f);
    ImGui::SetNextWindowPos(ImVec2(vpos.x + 12.0f, vpos.y + 12.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_Always);
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_AlwaysAutoResize;

    if (ImGui::Begin("Stats", &show_stats_, flags)) {
      const auto stats = player.statsSnapshot();
      const std::string hwdec = player.hwdecCurrent();
      const std::string hwdec_req = player.hwdecRequested();
      const int hwdec_extra_frames_req = player.hwdecExtraFramesRequested();
      const std::string fbo_format_req = player.fboFormatRequested();
      const std::string pixfmt = player.videoPixelFormat();

      ImGui::TextDisabled("Video");
      ImGui::Separator();
      ImGui::Text("Decoded: %u x %u", player.videoWidth(), player.videoHeight());
      if (!pixfmt.empty())
        ImGui::Text("pixelformat:      %s", pixfmt.c_str());
      ImGui::Text("hwdec (requested): %s", hwdec_req.empty() ? "(unset)" : hwdec_req.c_str());
      ImGui::Text("hwdec-current:     %s", hwdec.empty() ? "(none)" : hwdec.c_str());

      if (cb.set_hwdec) {
        ImGui::Spacing();
        ImGui::TextDisabled("hwdec override");
        ImGui::Separator();

        if (hwdec_buf_[0] == 0 && hwdec_req.size() < hwdec_buf_.size()) {
          std::memset(hwdec_buf_.data(), 0, hwdec_buf_.size());
          std::memcpy(hwdec_buf_.data(), hwdec_req.data(), hwdec_req.size());
        }

        ImGui::SetNextItemWidth(220);
        if (ImGui::InputTextWithHint("##hwdec", "auto / vaapi / nvdec / (empty)", hwdec_buf_.data(), hwdec_buf_.size(),
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
          cb.set_hwdec(std::string(hwdec_buf_.data()));
          dirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply")) {
          cb.set_hwdec(std::string(hwdec_buf_.data()));
          dirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Auto")) {
          std::memset(hwdec_buf_.data(), 0, hwdec_buf_.size());
          std::memcpy(hwdec_buf_.data(), "auto", 4);
          cb.set_hwdec("auto");
          dirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("nvdec")) {
          std::memset(hwdec_buf_.data(), 0, hwdec_buf_.size());
          std::memcpy(hwdec_buf_.data(), "nvdec", 5);
          cb.set_hwdec("nvdec");
          dirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("nvdec-copy")) {
          std::memset(hwdec_buf_.data(), 0, hwdec_buf_.size());
          std::memcpy(hwdec_buf_.data(), "nvdec-copy", 10);
          cb.set_hwdec("nvdec-copy");
          dirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) {
          std::memset(hwdec_buf_.data(), 0, hwdec_buf_.size());
          std::memcpy(hwdec_buf_.data(), "no", 2);
          cb.set_hwdec("no");
          dirty_ = true;
        }
      }

      if (cb.set_hwdec_extra_frames || cb.set_fbo_format) {
        ImGui::Spacing();
        ImGui::TextDisabled("buffers");
        ImGui::Separator();

        if (cb.set_hwdec_extra_frames) {
          if (hwdec_extra_frames_buf_[0] == 0) {
            const std::string v = std::to_string(hwdec_extra_frames_req);
            if (v.size() < hwdec_extra_frames_buf_.size()) {
              std::memset(hwdec_extra_frames_buf_.data(), 0, hwdec_extra_frames_buf_.size());
              std::memcpy(hwdec_extra_frames_buf_.data(), v.data(), v.size());
            }
          }
          ImGui::SetNextItemWidth(80);
          if (ImGui::InputTextWithHint("##hwdec_extra_frames", "extra", hwdec_extra_frames_buf_.data(),
                                       hwdec_extra_frames_buf_.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
            const int v = std::atoi(hwdec_extra_frames_buf_.data());
            cb.set_hwdec_extra_frames(v);
            dirty_ = true;
          }
          ImGui::SameLine();
          ImGui::Text("hwdec-extra-frames");
          ImGui::SameLine();
          if (ImGui::SmallButton("0")) {
            std::memset(hwdec_extra_frames_buf_.data(), 0, hwdec_extra_frames_buf_.size());
            std::memcpy(hwdec_extra_frames_buf_.data(), "0", 1);
            cb.set_hwdec_extra_frames(0);
            dirty_ = true;
          }
          ImGui::SameLine();
          if (ImGui::SmallButton("2")) {
            std::memset(hwdec_extra_frames_buf_.data(), 0, hwdec_extra_frames_buf_.size());
            std::memcpy(hwdec_extra_frames_buf_.data(), "2", 1);
            cb.set_hwdec_extra_frames(2);
            dirty_ = true;
          }
          ImGui::SameLine();
          if (ImGui::SmallButton("4")) {
            std::memset(hwdec_extra_frames_buf_.data(), 0, hwdec_extra_frames_buf_.size());
            std::memcpy(hwdec_extra_frames_buf_.data(), "4", 1);
            cb.set_hwdec_extra_frames(4);
            dirty_ = true;
          }
        }

        if (cb.set_fbo_format) {
          if (fbo_format_buf_[0] == 0 && fbo_format_req.size() < fbo_format_buf_.size()) {
            std::memset(fbo_format_buf_.data(), 0, fbo_format_buf_.size());
            std::memcpy(fbo_format_buf_.data(), fbo_format_req.data(), fbo_format_req.size());
          }
          ImGui::SetNextItemWidth(120);
          if (ImGui::InputTextWithHint("##fbo_format", "rgba8/rgba16f", fbo_format_buf_.data(),
                                       fbo_format_buf_.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
            cb.set_fbo_format(std::string(fbo_format_buf_.data()));
            dirty_ = true;
          }
          ImGui::SameLine();
          ImGui::Text("fbo-format");
          ImGui::SameLine();
          if (ImGui::SmallButton("rgba8")) {
            std::memset(fbo_format_buf_.data(), 0, fbo_format_buf_.size());
            std::memcpy(fbo_format_buf_.data(), "rgba8", 5);
            cb.set_fbo_format("rgba8");
            dirty_ = true;
          }
          ImGui::SameLine();
          if (ImGui::SmallButton("rgba16f")) {
            std::memset(fbo_format_buf_.data(), 0, fbo_format_buf_.size());
            std::memcpy(fbo_format_buf_.data(), "rgba16f", 6);
            cb.set_fbo_format("rgba16f");
            dirty_ = true;
          }
        }
      }

      ImGui::Spacing();
      ImGui::TextDisabled("Render");
      ImGui::Separator();
      ImGui::Text("calls=%llu updates=%llu frames=%llu failures=%llu", static_cast<unsigned long long>(stats.renderCalls),
                  static_cast<unsigned long long>(stats.renderUpdates), static_cast<unsigned long long>(stats.renderFrames),
                  static_cast<unsigned long long>(stats.renderFailures));
      ImGui::Text("frame ms: last=%.2f ema=%.2f max=%.2f", stats.lastFrameTotalMs, stats.emaFrameTotalMs,
                  stats.maxFrameTotalMs);
      ImGui::Text("stutter(>=50ms): %llu (last=%.2fms)", static_cast<unsigned long long>(stats.stutterCount),
                  stats.lastStutterMs);

      ImGui::Spacing();
      ImGui::TextDisabled("SHM readback");
      ImGui::Separator();
      ImGui::Text("written=%llu issued=%llu mapped=%llu", static_cast<unsigned long long>(stats.shmWritten),
                  static_cast<unsigned long long>(stats.shmReadbacksIssued),
                  static_cast<unsigned long long>(stats.shmReadbacksMapped));
      ImGui::Text("last dims: %u x %u", stats.lastShmWidth, stats.lastShmHeight);
      ImGui::Text("issue ms: %.2f   map+write ms: last=%.2f ema=%.2f", stats.lastShmIssueMs, stats.lastShmMapWriteMs,
                  stats.emaShmMapWriteMs);
      ImGui::Text("skips: noSink=%llu interval=%llu target=%llu sinkCfg=%llu busy=%llu",
                  static_cast<unsigned long long>(stats.shmSkipNoSink),
                  static_cast<unsigned long long>(stats.shmSkipInterval),
                  static_cast<unsigned long long>(stats.shmSkipTarget),
                  static_cast<unsigned long long>(stats.shmSkipSinkConfig),
                  static_cast<unsigned long long>(stats.shmSkipReadbackBusy));

      ImGui::Spacing();
      ImGui::TextDisabled("GPU (estimated, our targets only)");
      ImGui::Separator();
      ImGui::Text("video target: %.1f MiB", bytes_to_mib(stats.estVideoTargetBytes));
      ImGui::Text("downsample:  %.1f MiB", bytes_to_mib(stats.estDownsampleTargetBytes));
      ImGui::Text("readback PBO: %.1f MiB", bytes_to_mib(stats.estReadbackPboBytes));
      ImGui::Text("total:       %.1f MiB", bytes_to_mib(stats.estTotalBytes));
    }
    ImGui::End();
  }

  if (show_playlist_ && !playlist.empty()) {
    const float list_w = std::min(420.0f, vsize.x - 24.0f);
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::SetNextWindowPos(ImVec2(vpos.x + vsize.x - list_w - 12.0f, vpos.y + 12.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(list_w, std::min(520.0f, vsize.y - 24.0f)), ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::Begin("##implayer_playlist", nullptr, flags)) {
      ImGui::Text("Playlist (%d)", static_cast<int>(playlist.size()));
      ImGui::Separator();
      for (int i = 0; i < static_cast<int>(playlist.size()); ++i) {
        const bool selected = (i == playlist_index);
        std::string label = selected ? ("> " + playlist[i]) : playlist[i];
        if (ImGui::Selectable(label.c_str(), selected)) {
          if (cb.playlist_select)
            cb.playlist_select(i);
          dirty_ = true;
        }
      }
    }
    ImGui::End();
  }

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace f8::implayer
