#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

extern "C" {
#include <mpv/client.h>
#include <mpv/render.h>
#include <mpv/render_gl.h>
}

namespace f8::implayer {

class VideoSharedMemorySink;

class MpvPlayer {
 public:
  struct VideoConfig {
    bool offline = false;
    std::uint32_t videoShmMaxWidth = 1920;
    std::uint32_t videoShmMaxHeight = 1080;
    double videoShmMaxFps = 30.0;
  };

  using TimeCallback = std::function<void(double positionSeconds, double durationSeconds)>;
  using PlayingCallback = std::function<void(bool playing)>;
  using FinishedCallback = std::function<void()>;

  MpvPlayer(const VideoConfig& config, TimeCallback timeCallback, PlayingCallback playingCallback,
            FinishedCallback finishedCallback);
  ~MpvPlayer();

  bool openMedia(const std::string& source);
  bool play();
  void pause();
  void stop();
  void setVolume(double volume01);
  void seek(double positionSeconds);
  void setSharedMemorySink(std::shared_ptr<VideoSharedMemorySink> sink);
  void setVideoShmMaxSize(std::uint32_t max_width, std::uint32_t max_height);
  void setVideoShmMaxFps(double max_fps);
  std::uint32_t videoShmMaxWidth() const;
  std::uint32_t videoShmMaxHeight() const;
  double videoShmMaxFps() const;

  bool initializeGl();
  void shutdownGl();
  bool renderVideoFrame();

  unsigned videoWidth() const { return videoWidth_.load(std::memory_order_relaxed); }
  unsigned videoHeight() const { return videoHeight_.load(std::memory_order_relaxed); }
  unsigned videoFboId() const { return videoFboId_; }
  double positionSeconds() const { return lastPositionSeconds_.load(std::memory_order_relaxed); }
  double durationSeconds() const { return lastDurationSeconds_.load(std::memory_order_relaxed); }

 private:
  void initializeMpv();
  void startEventThread();
  void stopEventThread();
  void eventLoop();
  void processEvent(const mpv_event& event);
  void handlePropertyChange(const mpv_event_property& property);
  void handleVideoReconfig();
  std::pair<unsigned, unsigned> targetDimensions(unsigned width, unsigned height) const;
  std::shared_ptr<VideoSharedMemorySink> sharedSink() const;
  bool ensureVideoTargets(unsigned width, unsigned height);
  bool ensureDownsampleTargets(unsigned width, unsigned height);
  bool ensureReadbackPbos(std::size_t bytes);
  void destroyReadbackPbos();
  bool copyFrameToSharedMemory(unsigned width, unsigned height, double& copyMs);
  bool shouldCopyToSharedMemory(std::chrono::steady_clock::time_point now) const;

  static void HandleMpvWakeup(void* userdata);
  static void HandleRenderUpdate(void* userdata);
  static void* GetProcAddress(void* ctx, const char* name);

  VideoConfig config_;
  std::atomic<std::uint32_t> shm_max_width_{0};
  std::atomic<std::uint32_t> shm_max_height_{0};
  std::atomic<double> shm_max_fps_{0.0};
  std::atomic<double> shm_frame_interval_s_{0.0};
  TimeCallback timeCallback_;
  PlayingCallback playingCallback_;
  FinishedCallback finishedCallback_;

  mpv_handle* mpv_ = nullptr;
  mpv_render_context* renderContext_ = nullptr;

  std::thread eventThread_;
  std::atomic<bool> stopRequested_{false};
  std::atomic<bool> renderUpdatePending_{false};
  std::condition_variable eventCv_;
  std::mutex eventMutex_;
  bool eventNotified_ = false;

  mutable std::mutex sinkMutex_;
  std::shared_ptr<VideoSharedMemorySink> sink_;
  std::chrono::steady_clock::time_point lastShmFrameTime_{};

  std::atomic<unsigned> videoWidth_{0};
  std::atomic<unsigned> videoHeight_{0};
  std::atomic<double> lastPositionSeconds_{0.0};
  std::atomic<double> lastDurationSeconds_{0.0};

  unsigned int videoTextureId_ = 0;
  unsigned int videoFboId_ = 0;
  unsigned videoTexWidth_ = 0;
  unsigned videoTexHeight_ = 0;
  unsigned int downsampleTextureId_ = 0;
  unsigned int downsampleFboId_ = 0;
  unsigned downsampleWidth_ = 0;
  unsigned downsampleHeight_ = 0;
  static constexpr int kReadbackPboCount = 2;
  unsigned int readbackPbos_[kReadbackPboCount] = {};
  bool readbackPboInUse_[kReadbackPboCount] = {};
  std::size_t readbackPboSize_ = 0;
  int readbackPboIndex_ = 0;
  bool glInitialized_ = false;
};

}  // namespace f8::implayer
