#include <iostream>
#include <memory>
#include <string>

#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL_main.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include "f8cppsdk/describe_builtins.h"
#include "implayer_service.h"

namespace {

struct AppState {
  std::unique_ptr<f8::implayer::ImPlayerService> service;
};

}  // namespace

extern "C" SDL_AppResult SDLCALL SDL_AppInit(void** appstate, int argc, char* argv[]) {
  if (appstate)
    *appstate = nullptr;

  try {
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  } catch (...) {}
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);

  cxxopts::Options options("f8implayer_service", "F8 C++ IM Player service");
  options.add_options()("describe", "Print service spec JSON and exit")(
      "service-id", "Service instance id (required unless --describe)",
      cxxopts::value<std::string>()->default_value(""))(
      "nats-url", "NATS server URL", cxxopts::value<std::string>()->default_value("nats://127.0.0.1:4222"))(
      "media", "Open media URL/path on startup", cxxopts::value<std::string>()->default_value(""))("help", "Show help");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return SDL_APP_SUCCESS;
  }

  if (result.count("describe")) {
    const auto payload =
        f8::cppsdk::normalize_describe_with_builtin_state_fields(f8::implayer::ImPlayerService::describe());
    std::cout << payload.dump(1) << "\n";
    return SDL_APP_SUCCESS;
  }

  const std::string service_id = result["service-id"].as<std::string>();
  if (service_id.empty()) {
    std::cerr << "Missing --service-id\n";
    return SDL_APP_FAILURE;
  }

  f8::implayer::ImPlayerService::Config cfg;
  cfg.service_id = service_id;
  cfg.nats_url = result["nats-url"].as<std::string>();
  cfg.initial_media_url = result["media"].as<std::string>();

  try {
    auto state = std::make_unique<AppState>();
    state->service = std::make_unique<f8::implayer::ImPlayerService>(cfg);
    if (!state->service->start()) {
      spdlog::error("implayer service start failed");
      return SDL_APP_FAILURE;
    }
    *appstate = state.release();
    return SDL_APP_CONTINUE;
  } catch (const std::exception& e) {
    spdlog::critical("implayer init failed: {}", e.what());
    return SDL_APP_FAILURE;
  } catch (...) {
    spdlog::critical("implayer init failed: unknown error");
    return SDL_APP_FAILURE;
  }
}

extern "C" SDL_AppResult SDLCALL SDL_AppIterate(void* appstate) {
  auto* state = static_cast<AppState*>(appstate);
  if (!state || !state->service)
    return SDL_APP_FAILURE;
  state->service->tick();
  return state->service->running() ? SDL_APP_CONTINUE : SDL_APP_SUCCESS;
}

extern "C" SDL_AppResult SDLCALL SDL_AppEvent(void* appstate, SDL_Event* event) {
  auto* state = static_cast<AppState*>(appstate);
  if (!state || !state->service || !event)
    return SDL_APP_FAILURE;
  state->service->processSdlEvent(*event);
  return SDL_APP_CONTINUE;
}

extern "C" void SDLCALL SDL_AppQuit(void* appstate, SDL_AppResult /*result*/) {
  auto* state = static_cast<AppState*>(appstate);
  if (!state)
    return;
  if (state->service)
    state->service->stop();
  delete state;
}
