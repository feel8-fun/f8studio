#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "f8cppsdk/describe_builtins.h"
#include "screencap_service.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("f8screencap_service", "F8 screen capture (platform backend) -> video SHM service");
  options.add_options()("describe", "Print service spec JSON and exit")(
      "service-id", "Service instance id (required unless --describe)", cxxopts::value<std::string>()->default_value(""))(
      "nats-url", "NATS server URL", cxxopts::value<std::string>()->default_value("nats://127.0.0.1:4222"))(
      "shm-bytes", "Video SHM bytes", cxxopts::value<std::size_t>()->default_value(std::to_string(f8::cppsdk::shm::kDefaultVideoShmBytes)))(
      "shm-slots", "Video SHM slots", cxxopts::value<std::uint32_t>()->default_value(std::to_string(f8::cppsdk::shm::kDefaultVideoShmSlots)))(
      "fps", "Capture FPS", cxxopts::value<double>()->default_value("30.0"))(
      "mode", "Capture mode (display|window|region)", cxxopts::value<std::string>()->default_value("display"))(
      "display-id", "Display id (0..N-1)", cxxopts::value<int>()->default_value("0"))(
      "window-id", "Window id (backend-specific, e.g. win32:hwnd:0x... or x11:win:0x...)", cxxopts::value<std::string>()->default_value(""))(
      "region", "Region as x,y,w,h (mode=region)", cxxopts::value<std::string>()->default_value(""))(
      "scale", "Scale as w,h (optional; 0,0 disables)", cxxopts::value<std::string>()->default_value(""))("help", "Show help");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  if (result.count("describe")) {
    const auto payload =
        f8::cppsdk::normalize_describe_with_builtin_state_fields(f8::screencap::ScreenCapService::describe());
    std::cout << payload.dump(1) << "\n";
    return 0;
  }

  try {
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  } catch (...) {
  }
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);
  spdlog::info("starting f8screencap_service");

  std::signal(SIGINT, &on_signal);
  std::signal(SIGTERM, &on_signal);

  const std::string service_id = result["service-id"].as<std::string>();
  if (service_id.empty()) {
    std::cerr << "Missing --service-id\n";
    return 2;
  }

  f8::screencap::ScreenCapService::Config cfg;
  cfg.service_id = service_id;
  cfg.nats_url = result["nats-url"].as<std::string>();
  cfg.video_shm_bytes = result["shm-bytes"].as<std::size_t>();
  cfg.video_shm_slots = result["shm-slots"].as<std::uint32_t>();
  cfg.mode = result["mode"].as<std::string>();
  cfg.fps = result["fps"].as<double>();
  cfg.display_id = result["display-id"].as<int>();
  cfg.window_id = result["window-id"].as<std::string>();
  cfg.region_csv = result["region"].as<std::string>();
  cfg.scale_csv = result["scale"].as<std::string>();

  f8::screencap::ScreenCapService svc(cfg);
  if (!svc.start()) {
    spdlog::error("screencap service start failed");
    return 1;
  }

  while (!g_stop.load(std::memory_order_acquire) && svc.running()) {
    svc.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  svc.stop();
  return 0;
}
