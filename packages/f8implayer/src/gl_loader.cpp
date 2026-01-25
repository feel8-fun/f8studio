#include "gl_loader.h"

#include <atomic>
#include <mutex>

#define SDL_MAIN_HANDLED 1
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <spdlog/spdlog.h>

namespace f8::implayer {

namespace {
std::atomic<bool> g_glad_loaded{false};
std::mutex g_glad_mu;
}  // namespace

bool EnsureOpenGLFunctionsLoaded() {
  if (g_glad_loaded.load(std::memory_order_acquire)) {
    return true;
  }

  std::lock_guard<std::mutex> lock(g_glad_mu);
  if (g_glad_loaded.load(std::memory_order_relaxed)) {
    return true;
  }

  if (!SDL_GL_GetCurrentContext()) {
    spdlog::debug("OpenGL loader init skipped: no current SDL GL context.");
    return false;
  }

  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
    spdlog::error("gladLoadGLLoader failed");
    return false;
  }
  g_glad_loaded.store(true, std::memory_order_release);
  return true;
}

}  // namespace f8::implayer
