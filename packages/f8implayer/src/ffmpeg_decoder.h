#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace f8::implayer {

class VideoSharedMemorySink;

class FfmpegDecoder {
 public:
  struct Config {
    double max_fps = 30.0;  // <=0 means unlimited
  };

  FfmpegDecoder(Config cfg, std::shared_ptr<VideoSharedMemorySink> sink);
  ~FfmpegDecoder();

  bool open(const std::string& url, std::string& err);
  void close();

  void play();
  void pause();
  void stop();
  bool seek(double position_seconds, std::string& err);

  bool playing() const { return playing_.load(std::memory_order_acquire); }
  double position_seconds() const { return position_s_.load(std::memory_order_acquire); }
  double duration_seconds() const { return duration_s_.load(std::memory_order_acquire); }
  std::string url() const;
  std::string last_error() const;

 private:
  void worker();
  void set_error(std::string err);

  Config cfg_;
  std::shared_ptr<VideoSharedMemorySink> sink_;

  std::atomic<bool> stop_{false};
  std::atomic<bool> playing_{false};
  std::atomic<double> position_s_{0.0};
  std::atomic<double> duration_s_{0.0};

  mutable std::mutex mu_;
  std::string url_;
  std::string last_error_;

  std::thread thread_;
  std::atomic<bool> reopen_{false};
  std::atomic<double> seek_req_s_{-1.0};
};

}  // namespace f8::implayer

