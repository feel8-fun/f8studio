#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "f8cppsdk/describe_builtins.h"
#include "f8cppsdk/runtime_cxxopts.h"
#include "tracking_service.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("f8cvkit_tracking_service", "CVKit tracking service (OpenCV contrib tracking)");
  options.add_options()("describe", "Print service spec JSON and exit")(
      "service-id", "Service instance id (required unless --describe)", cxxopts::value<std::string>()->default_value(""))(
      "tracker-kind", "Tracker backend (csrt|kcf|mil|nano|vit)",
      cxxopts::value<std::string>()->default_value("csrt"))(
      "model-dir", "Directory for tracker model files used by nano/vit",
      cxxopts::value<std::string>()->default_value(""))(
      "auto-download-models", "Auto-download missing tracker model files when needed",
      cxxopts::value<bool>()->default_value("true")->implicit_value("true"))(
      "max-tracking-fps", "Maximum tracker update rate (0 = unlimited)",
      cxxopts::value<double>()->default_value("30"))(
      "stop-cooldown-ms", "Cooldown after stopTracking (ignore initBox for N ms)",
      cxxopts::value<int>()->default_value("1000"))(
      "help", "Show help");
  f8::cppsdk::add_runtime_backend_options(options);

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  if (result.count("describe")) {
    const auto payload =
        f8::cppsdk::normalize_describe_with_builtin_state_fields(f8::cvkit::tracking::TrackingService::describe());
    std::cout << payload.dump(1) << "\n";
    return 0;
  }

  try {
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  } catch (const std::exception& ex) {
    std::cerr << "logger init failed: " << ex.what() << "\n";
  } catch (...) {
    std::cerr << "logger init failed: unknown exception\n";
  }
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);

  std::signal(SIGINT, &on_signal);
  std::signal(SIGTERM, &on_signal);

  const std::string service_id = result["service-id"].as<std::string>();
  if (service_id.empty()) {
    std::cerr << "Missing --service-id\n";
    return 2;
  }

  f8::cppsdk::RuntimeBackendConfig runtime_backend;
  std::string runtime_error;
  if (!f8::cppsdk::read_runtime_backend_options(result, runtime_backend, runtime_error)) {
    std::cerr << runtime_error << "\n";
    return 2;
  }

  f8::cvkit::tracking::TrackingService::Config cfg;
  cfg.service_id = service_id;
  cfg.runtime_backend = runtime_backend;
  cfg.tracker_kind = result["tracker-kind"].as<std::string>();
  cfg.model_dir = result["model-dir"].as<std::string>();

  cfg.auto_download_models = result["auto-download-models"].as<bool>();
  cfg.max_tracking_fps = result["max-tracking-fps"].as<double>();
  cfg.stop_tracking_cooldown_ms = result["stop-cooldown-ms"].as<int>();

  f8::cvkit::tracking::TrackingService svc(cfg);
  if (!svc.start()) {
    spdlog::error("cvkit_tracking start failed");
    return 1;
  }

  while (!g_stop.load(std::memory_order_acquire) && svc.running()) {
    svc.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  svc.stop();
  return 0;
}
