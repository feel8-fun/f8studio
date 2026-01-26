#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <SDL3/SDL.h>

namespace f8::audiocap {

class WasapiLoopbackCapture final {
 public:
  struct Config {
    std::uint32_t dst_sample_rate = 48000;
    std::uint16_t dst_channels = 2;
  };

  using Callback = std::function<void(const float* interleaved_f32, std::uint32_t frames, std::int64_t ts_ms)>;

  explicit WasapiLoopbackCapture(Config cfg);
  ~WasapiLoopbackCapture();

  WasapiLoopbackCapture(const WasapiLoopbackCapture&) = delete;
  WasapiLoopbackCapture& operator=(const WasapiLoopbackCapture&) = delete;

  bool start(Callback cb, std::string& out_device_name, std::string& out_error);
  void stop();

  void set_paused(bool paused) { paused_.store(paused, std::memory_order_release); }

 private:
  void thread_main();

  Config cfg_;
  Callback cb_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> paused_{false};

  std::thread worker_;

  std::mutex init_mu_;
  std::condition_variable init_cv_;
  bool init_done_ = false;

  std::string device_name_;
  std::string error_;
};

}  // namespace f8::audiocap
