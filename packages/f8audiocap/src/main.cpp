#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <SDL3/SDL.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "audiocap_service.h"
#include "f8cppsdk/shm/sizing.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("f8audiocap_service", "F8 audio capture service (system mix / loopback)");
  options.add_options()("describe", "Print service spec JSON and exit")(
      "service-id", "Service instance id (required unless --describe)", cxxopts::value<std::string>()->default_value(""))(
      "nats-url", "NATS server URL", cxxopts::value<std::string>()->default_value("nats://127.0.0.1:4222"))(
      "list-devices", "List available recording devices and exit")(
      "backend", "Backend (auto|sdl|wasapi)", cxxopts::value<std::string>()->default_value("auto"))(
      "device", "Recording device selector (index or substring match)", cxxopts::value<std::string>()->default_value(""))(
      "shm-bytes", "Audio SHM bytes (0=auto)", cxxopts::value<std::size_t>()->default_value(std::to_string(f8::cppsdk::shm::kDefaultAudioShmBytes)))(
      "sample-rate", "Sample rate", cxxopts::value<std::uint32_t>()->default_value("48000"))(
      "channels", "Channels", cxxopts::value<std::uint16_t>()->default_value("2"))(
      "frames-per-chunk", "Frames per chunk", cxxopts::value<std::uint32_t>()->default_value("480"))(
      "chunk-count", "Chunk count", cxxopts::value<std::uint32_t>()->default_value("200"))(
      "mode", "Mode (capture|silence|sine)", cxxopts::value<std::string>()->default_value("capture"))(
      "tone-hz", "Sine tone Hz (mode=sine)", cxxopts::value<double>()->default_value("440.0"))(
      "gain", "Output gain 0..1", cxxopts::value<double>()->default_value("0.1"))("help", "Show help");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  if (result.count("describe")) {
    std::cout << f8::audiocap::AudioCapService::describe().dump(1) << "\n";
    return 0;
  }

  if (result.count("list-devices")) {
    if (!SDL_Init(SDL_INIT_AUDIO)) {
      std::cerr << "SDL_Init(SDL_INIT_AUDIO) failed: " << SDL_GetError() << "\n";
      return 1;
    }
    int count = 0;
    SDL_AudioDeviceID* devices = SDL_GetAudioRecordingDevices(&count);
    std::cout << "Recording devices (" << count << "):\n";
    for (int i = 0; i < count; ++i) {
      const SDL_AudioDeviceID id = devices[i];
      const char* name = SDL_GetAudioDeviceName(id);
      std::cout << "  [" << i << "] " << (name ? name : "(null)") << "\n";
    }
    if (devices) SDL_free(devices);
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
#if defined(_WIN32)
    std::cout << "\nWindows system mix capture uses WASAPI loopback. Use: --backend wasapi\n";
#endif
    return 0;
  }

  try {
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));
  } catch (...) {
  }
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);
  spdlog::info("starting f8audiocap_service");

  std::signal(SIGINT, &on_signal);
  std::signal(SIGTERM, &on_signal);

  const std::string service_id = result["service-id"].as<std::string>();
  if (service_id.empty()) {
    std::cerr << "Missing --service-id\n";
    return 2;
  }

  f8::audiocap::AudioCapService::Config cfg;
  cfg.service_id = service_id;
  cfg.nats_url = result["nats-url"].as<std::string>();
  cfg.audio_shm_bytes = result["shm-bytes"].as<std::size_t>();
  cfg.sample_rate = result["sample-rate"].as<std::uint32_t>();
  cfg.channels = result["channels"].as<std::uint16_t>();
  cfg.frames_per_chunk = result["frames-per-chunk"].as<std::uint32_t>();
  cfg.chunk_count = result["chunk-count"].as<std::uint32_t>();
  cfg.mode = result["mode"].as<std::string>();
  cfg.backend = result["backend"].as<std::string>();
  cfg.device = result["device"].as<std::string>();
  cfg.tone_hz = result["tone-hz"].as<double>();
  cfg.gain = result["gain"].as<double>();

  const std::size_t required = f8::cppsdk::shm::audio_required_bytes(
      cfg.sample_rate, cfg.channels, cfg.frames_per_chunk, cfg.chunk_count, f8::cppsdk::AudioSharedMemorySink::SampleFormat::kF32LE);
  const std::size_t recommended =
      f8::cppsdk::shm::audio_recommended_bytes(cfg.sample_rate, cfg.channels, cfg.frames_per_chunk, cfg.chunk_count,
                                               f8::cppsdk::AudioSharedMemorySink::SampleFormat::kF32LE);
  if (cfg.audio_shm_bytes == 0) {
    cfg.audio_shm_bytes = recommended;
  } else if (required != 0 && cfg.audio_shm_bytes < required) {
    std::cerr << "Audio SHM too small: --shm-bytes=" << cfg.audio_shm_bytes << " required>=" << required
              << " (try --shm-bytes=0 for auto)\n";
    return 2;
  }

  f8::audiocap::AudioCapService svc(cfg);
  if (!svc.start()) {
    spdlog::error("audiocap service start failed");
    return 1;
  }

  while (!g_stop.load(std::memory_order_acquire) && svc.running()) {
    svc.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  svc.stop();
  return 0;
}
