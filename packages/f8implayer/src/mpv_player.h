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

namespace f8::cppsdk {
class VideoSharedMemorySink;
}

namespace f8::implayer {

using VideoSharedMemorySink = ::f8::cppsdk::VideoSharedMemorySink;

class MpvPlayer {
 public:
  struct Stats {
    std::uint64_t renderCalls = 0;
    std::uint64_t renderUpdates = 0;
    std::uint64_t renderFrames = 0;
    std::uint64_t renderFailures = 0;

    double lastFrameTotalMs = 0.0;
    double emaFrameTotalMs = 0.0;
    double maxFrameTotalMs = 0.0;
    std::uint64_t stutterCount = 0;
    double lastStutterMs = 0.0;

    std::uint64_t shmWritten = 0;
    std::uint64_t shmSkipNoSink = 0;
    std::uint64_t shmSkipInterval = 0;
    std::uint64_t shmSkipTarget = 0;
    std::uint64_t shmSkipSinkConfig = 0;
    std::uint64_t shmSkipReadbackBusy = 0;
    std::uint64_t shmReadbacksIssued = 0;
    std::uint64_t shmReadbacksMapped = 0;

    unsigned lastShmWidth = 0;
    unsigned lastShmHeight = 0;
    double lastShmIssueMs = 0.0;
    double lastShmMapWriteMs = 0.0;
    double emaShmMapWriteMs = 0.0;
    double shmSinceLastWriteMs = 0.0;
    std::uint64_t lastShmWriteSteadyNs = 0;

    std::uint64_t estVideoTargetBytes = 0;
    std::uint64_t estDownsampleTargetBytes = 0;
    std::uint64_t estReadbackPboBytes = 0;
    std::uint64_t estTotalBytes = 0;
  };

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
  bool setHwdec(const std::string& hwdec);
  bool setHwdecExtraFrames(int extra_frames);
  bool setFboFormat(const std::string& fbo_format);
  void resetVideoOutput();
  void resetPlaybackState();
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
  Stats statsSnapshot() const;
  std::string hwdecCurrent() const;
  std::string hwdecRequested() const;
  int hwdecExtraFramesRequested() const;
  std::string fboFormatRequested() const;
  std::string videoPixelFormat() const;

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
  bool shouldCopyToSharedMemory(std::chrono::steady_clock::time_point now);

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

  // SHM rate-limiter state (render thread).
  mutable std::mutex shmRateMutex_;
  bool shm_due_initialized_ = false;
  std::chrono::steady_clock::time_point shm_next_due_{};
  std::atomic<bool> shm_rate_reset_{false};

  std::atomic<unsigned> videoWidth_{0};
  std::atomic<unsigned> videoHeight_{0};
  std::atomic<double> lastPositionSeconds_{0.0};
  std::atomic<double> lastDurationSeconds_{0.0};
  std::atomic<bool> eofReached_{false};

  mutable std::mutex statsMutex_;
  Stats stats_{};

  mutable std::mutex mpvPropsMutex_;
  std::string hwdecCurrent_;
  std::string hwdecRequested_;
  int hwdecExtraFramesRequested_ = 2;
  std::string fboFormatRequested_;
  std::string videoPixelFormat_;

  unsigned int videoTextureId_ = 0;
  unsigned int videoFboId_ = 0;
  unsigned videoTexWidth_ = 0;
  unsigned videoTexHeight_ = 0;
  unsigned int downsampleTextureId_ = 0;
  unsigned int downsampleFboId_ = 0;
  unsigned downsampleWidth_ = 0;
  unsigned downsampleHeight_ = 0;
  static constexpr int kReadbackPboCount = 3;
  unsigned int readbackPbos_[kReadbackPboCount] = {};
  bool readbackPboInUse_[kReadbackPboCount] = {};
  void* readbackFences_[kReadbackPboCount] = {};
  std::size_t readbackPboSize_ = 0;
  int readbackPboIndex_ = 0;
  bool glInitialized_ = false;
};

}  // namespace f8::implayer
