#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <SDL3/SDL.h>
#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/audio_shared_memory_sink.h"
#include "f8cppsdk/shm/audio.h"
#include "f8cppsdk/kv_store.h"
#include "f8cppsdk/nats_client.h"
#include "f8cppsdk/service_control_plane.h"
#include "f8cppsdk/service_control_plane_server.h"
#include "wasapi_loopback_capture.h"

namespace f8::audiocap {

class AudioCapService final : public f8::cppsdk::ServiceControlHandler {
 public:
  struct Config {
    std::string service_id;
    std::string service_class = "f8.audiocap";
    std::string nats_url = "nats://127.0.0.1:4222";

    std::size_t audio_shm_bytes = f8::cppsdk::shm::kDefaultAudioShmBytes;
    std::uint32_t sample_rate = 48000;
    std::uint16_t channels = 2;
    std::uint32_t frames_per_chunk = 480;
    std::uint32_t chunk_count = 200;
    std::string device = "";  // substring match (recording device); empty=default
    std::string backend = "auto";  // auto|sdl|wasapi (Windows: wasapi=system mix loopback)
    std::string mode = "capture";  // capture|silence|sine
    double tone_hz = 440.0;
    double gain = 0.1;
  };

  explicit AudioCapService(Config cfg);
  ~AudioCapService();

  bool start();
  void stop();
  bool running() const {
    return running_.load(std::memory_order_acquire) && !stop_requested_.load(std::memory_order_acquire);
  }

  void tick();

  bool is_active() const override { return active_.load(std::memory_order_acquire); }
  void on_activate(const nlohmann::json& meta) override;
  void on_deactivate(const nlohmann::json& meta) override;
  void on_set_active(bool active, const nlohmann::json& meta) override;
  bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                    const nlohmann::json& meta, std::string& error_code, std::string& error_message) override;
  bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                       std::string& error_message) override;
  bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                  nlohmann::json& result, std::string& error_code, std::string& error_message) override;

  static nlohmann::json describe();

 private:
  void set_active_local(bool active, const nlohmann::json& meta);
  void publish_static_state();
  void publish_dynamic_state();

  static std::string default_audio_shm_name(const std::string& service_id);

  Config cfg_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> active_{true};

  f8::cppsdk::NatsClient nats_;
  f8::cppsdk::KvStore kv_;
  std::unique_ptr<f8::cppsdk::ServiceControlPlaneServer> ctrl_;

  std::unique_ptr<f8::cppsdk::AudioSharedMemorySink> shm_;

  std::mutex state_mu_;
  std::unordered_map<std::string, nlohmann::json> published_state_;

  std::vector<float> chunk_buffer_;
  double phase_ = 0.0;
  std::int64_t last_write_ms_ = 0;
  std::int64_t last_state_pub_ms_ = 0;

  SDL_AudioStream* stream_ = nullptr;
  SDL_AudioDeviceID opened_device_ = 0;
  std::string opened_device_name_;
  std::vector<float> capture_tmp_;
  std::vector<float> capture_chunk_accum_;
  std::uint32_t capture_accum_frames_ = 0;

  static void SDLCALL on_audio_stream_put(void* userdata, SDL_AudioStream* stream, int additional_amount,
                                         int total_amount);
  void handle_audio_stream_put(SDL_AudioStream* stream, int additional_amount, int total_amount);

  void handle_captured_interleaved_f32(const float* interleaved, std::uint32_t frames, std::int64_t ts_ms);

  std::unique_ptr<WasapiLoopbackCapture> wasapi_;
};

}  // namespace f8::audiocap
