#include "implayer_gui.h"

#include <algorithm>
#include <cstring>

#include <imgui.h>

#include "imgui_backends/imgui_impl_opengl3.h"
#include "imgui_backends/imgui_impl_sdl3.h"
#include "mpv_player.h"

namespace f8::implayer {

ImPlayerGui::ImPlayerGui() { url_buf_.fill(0); }

ImPlayerGui::~ImPlayerGui() { stop(); }

bool ImPlayerGui::start(SDL_Window* window, SDL_GLContext gl_context) {
  if (started_) return true;
  if (!window || !gl_context) return false;

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
  if (!started_) return;
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  started_ = false;
}

void ImPlayerGui::processEvent(SDL_Event* ev) {
  if (!started_ || !ev) return;
  ImGui_ImplSDL3_ProcessEvent(ev);
  dirty_ = true;
}

void ImPlayerGui::renderOverlay(const MpvPlayer& player, const Callbacks& cb, const std::string& last_error) {
  if (!started_) return;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowBgAlpha(0.65f);
  ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_Always);

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize;
  if (ImGui::Begin("IM Player", nullptr, flags)) {
    ImGui::Text("Video: %ux%u", player.videoWidth(), player.videoHeight());
    const double dur = std::max(0.0, player.durationSeconds());
    const double pos = std::clamp(player.positionSeconds(), 0.0, dur > 0.0 ? dur : player.positionSeconds());
    ImGui::Text("Time: %.2fs / %.2fs", pos, dur);

    ImGui::Separator();

    ImGui::TextUnformatted("Open:");
    ImGui::SetNextItemWidth(-1);
    bool edited = ImGui::InputText("##url", url_buf_.data(), url_buf_.size());
    if (edited) dirty_ = true;
    if (ImGui::Button("Open") && cb.open) {
      cb.open(std::string(url_buf_.data()));
      dirty_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Play") && cb.play) {
      cb.play();
      dirty_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause") && cb.pause) {
      cb.pause();
      dirty_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop") && cb.stop) {
      cb.stop();
      dirty_ = true;
    }

    float seek_pos = static_cast<float>(pos);
    float seek_max = dur > 0.0 ? static_cast<float>(dur) : 0.0f;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##seek", &seek_pos, 0.0f, seek_max, "pos=%.2fs", ImGuiSliderFlags_AlwaysClamp)) {
      dirty_ = true;
    }
    if (ImGui::IsItemDeactivatedAfterEdit() && cb.seek) {
      cb.seek(static_cast<double>(seek_pos));
      dirty_ = true;
    }

    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##volume", &volume01_, 0.0f, 1.0f, "vol=%.2f")) {
      if (cb.set_volume) cb.set_volume(static_cast<double>(volume01_));
      dirty_ = true;
    }

    if (!last_error.empty()) {
      ImGui::Separator();
      ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f), "Error: %s", last_error.c_str());
    }

    ImGui::Separator();
    ImGui::TextDisabled("Keys: Space=Play/Pause, Left/Right=Seek, Up/Down=Volume");
  }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace f8::implayer
