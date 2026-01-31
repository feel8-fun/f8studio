#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "demo_service.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("f8sdk_demo_service", "F8 C++ SDK demo service (capabilities + ServiceBus template)");
  options.add_options()("service-id", "Service instance id", cxxopts::value<std::string>()->default_value("demo"))(
      "nats-url", "NATS server URL", cxxopts::value<std::string>()->default_value("nats://127.0.0.1:4222"))(
      "help", "Show help");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  try {
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  } catch (...) {
  }
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);

  std::signal(SIGINT, &on_signal);
  std::signal(SIGTERM, &on_signal);

  f8::sdk_demo::DemoService::Config cfg;
  cfg.service_id = result["service-id"].as<std::string>();
  cfg.nats_url = result["nats-url"].as<std::string>();

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

