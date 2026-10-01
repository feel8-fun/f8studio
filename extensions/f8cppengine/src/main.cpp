#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "f8cppengine/cppengine_service.h"
#include "f8cppsdk/runtime_cxxopts.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("f8cppengine_service", "F8 C++ Engine service");
  options.add_options()("service-id", "Service instance id", cxxopts::value<std::string>()->default_value("engine_cpp"))(
      "describe", "Output the service description in JSON format")("help", "Show help");
  f8::cppsdk::add_runtime_backend_options(options);

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  try {
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  } catch (const std::exception&) {
  }
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);

  f8::cppsdk::RuntimeBackendConfig runtime_backend;
  std::string runtime_error;
  if (!f8::cppsdk::read_runtime_backend_options(result, runtime_backend, runtime_error)) {
    std::cerr << runtime_error << "\n";
    return 2;
  }

  f8::cppengine::CppEngineService::Config cfg;
  cfg.service_id = result["service-id"].as<std::string>();
  cfg.runtime_backend = runtime_backend;

  f8::cppengine::CppEngineService svc(cfg);
  if (result.count("describe")) {
    std::cout << svc.describe_json().dump(2) << "\n";
    return 0;
  }

  std::signal(SIGINT, &on_signal);
  std::signal(SIGTERM, &on_signal);

  if (!svc.start()) {
    spdlog::error("cppengine start failed");
    return 1;
  }

  while (!g_stop.load(std::memory_order_acquire) && svc.running()) {
    svc.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  svc.stop();
  return 0;
}
