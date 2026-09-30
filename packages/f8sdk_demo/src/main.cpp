#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "f8cppsdk/runtime_cxxopts.h"
#include "demo_service.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int run(int argc, char** argv) {
  cxxopts::Options options("f8sdk_demo_service", "F8 C++ SDK demo service (capabilities + ServiceBus template)");
  options.add_options()("service-id", "Service instance id", cxxopts::value<std::string>()->default_value("demo"))(
      "help", "Show help");
  f8::cppsdk::add_runtime_backend_options(options);

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);

  std::signal(SIGINT, &on_signal);
  std::signal(SIGTERM, &on_signal);

  f8::cppsdk::RuntimeBackendConfig runtime_backend;
  std::string runtime_error;
  if (!f8::cppsdk::read_runtime_backend_options(result, runtime_backend, runtime_error)) {
    std::cerr << runtime_error << "\n";
    return 2;
  }

  f8::sdk_demo::DemoService::Config cfg;
  cfg.service_id = result["service-id"].as<std::string>();
  cfg.runtime_backend = runtime_backend;

  f8::sdk_demo::DemoService svc(cfg);
  if (!svc.start()) {
    spdlog::error("sdk_demo start failed");
    return 1;
  }

  while (!g_stop.load(std::memory_order_acquire) && svc.running()) {
    svc.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  svc.stop();
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const cxxopts::exceptions::exception& error) {
    std::cerr << "Invalid command line: " << error.what() << "\nUse --help for supported options.\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "SDK demo failed: " << error.what() << "\n";
    return 1;
  }
}
