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
#include "video_stab_service.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("f8cvkit_video_stab_service", "CVKit realtime video stabilizer service");
  options.add_options()("describe", "Print service spec JSON and exit")(
      "service-id", "Service instance id (required unless --describe)", cxxopts::value<std::string>()->default_value(""))(
      "nats-url", "NATS server URL", cxxopts::value<std::string>()->default_value("nats://127.0.0.1:4222"))(
      "help", "Show help");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  if (result.count("describe")) {
    const auto payload = f8::cppsdk::normalize_describe_with_builtin_state_fields(
        f8::cvkit::video_stab::VideoStabService::describe());
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

  f8::cvkit::video_stab::VideoStabService::Config cfg;
  cfg.service_id = service_id;
  cfg.nats_url = result["nats-url"].as<std::string>();

  f8::cvkit::video_stab::VideoStabService svc(cfg);
  if (!svc.start()) {
    spdlog::error("cvkit_video_stab start failed");
    return 1;
  }

  while (!g_stop.load(std::memory_order_acquire) && svc.running()) {
    svc.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  svc.stop();
  return 0;
}
