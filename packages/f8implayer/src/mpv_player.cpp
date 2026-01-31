#include "mpv_player.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <spdlog/spdlog.h>

#include "gl_loader.h"
#include "f8cppsdk/video_shared_memory_sink.h"

namespace f8::implayer {

namespace {

constexpr double kDefaultVolumePercent = 100.0;

double Clamp01(double value) {
  return std::clamp(value, 0.0, 1.0);
}

void LogMpvMessage(const mpv_event_log_message* log) {
  if (!log || !log->text)
    return;
  std::string_view text(log->text);
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
    text.remove_suffix(1);
  const char* level = log->level ? log->level : "";
  if (std::strcmp(level, "fatal") == 0)
    spdlog::critical("[mpv]{}", text);
  else if (std::strcmp(level, "error") == 0)
    spdlog::error("[mpv]{}", text);
  else if (std::strcmp(level, "warn") == 0 || std::strcmp(level, "warning") == 0)
    spdlog::warn("[mpv]{}", text);
  else if (std::strcmp(level, "info") == 0 || std::strcmp(level, "status") == 0)
    spdlog::info("[mpv]{}", text);
  else
    spdlog::debug("[mpv]{}", text);
}

}  // namespace

MpvPlayer::MpvPlayer(const VideoConfig& config, TimeCallback timeCallback, PlayingCallback playingCallback,
                     FinishedCallback finishedCallback)
    : config_(config),
      timeCallback_(std::move(timeCallback)),
      playingCallback_(std::move(playingCallback)),
      finishedCallback_(std::move(finishedCallback)) {
  shm_max_width_.store(config_.videoShmMaxWidth, std::memory_order_relaxed);
  shm_max_height_.store(config_.videoShmMaxHeight, std::memory_order_relaxed);
  shm_max_fps_.store(config_.videoShmMaxFps, std::memory_order_relaxed);
  shm_frame_interval_s_.store(config_.videoShmMaxFps > 0.0 ? (1.0 / config_.videoShmMaxFps) : 0.0,
                              std::memory_order_relaxed);
  initializeMpv();
  startEventThread();
}

MpvPlayer::~MpvPlayer() {
  shutdownGl();
  stopEventThread();
  if (renderContext_) {
    mpv_render_context_set_update_callback(renderContext_, nullptr, nullptr);
    mpv_render_context_free(renderContext_);
    renderContext_ = nullptr;
  }
  if (mpv_) {
    mpv_set_wakeup_callback(mpv_, nullptr, nullptr);
    mpv_terminate_destroy(mpv_);
    mpv_ = nullptr;
  }
}

void MpvPlayer::initializeMpv() {
  mpv_ = mpv_create();
  if (!mpv_)
    throw std::runtime_error("mpv_create failed");

  mpv_set_option_string(mpv_, "terminal", "no");
  mpv_set_option_string(mpv_, "msg-level", "all=v");
  mpv_set_option_string(mpv_, "config", "yes");
  mpv_set_option_string(mpv_, "load-scripts", "yes");
  // Keep the file open at EOF so seeking still works after playback finishes.
  mpv_set_option_string(mpv_, "keep-open", "yes");
  // Enable ytdl_hook.lua (youtube-dl / yt-dlp) when available; mpv will auto-detect the binary.
  mpv_set_option_string(mpv_, "ytdl", "yes");
  mpv_set_option_string(mpv_, "ytdl-format", "best");
#if defined(_WIN32)
  mpv_set_option_string(mpv_, "vo", "libmpv");
  // Prefer Windows native audio output. Some builds default to "null" ao when probing fails.
  mpv_set_option_string(mpv_, "ao", "wasapi");
#endif
  mpv_set_option_string(mpv_, "mute", "no");
  mpv_set_option_string(mpv_, "volume", "100");
  mpv_set_option_string(mpv_, "hwdec", "auto-safe");
  mpv_set_option_string(mpv_, "video-timing-offset", "0");

  if (int err = mpv_initialize(mpv_); err < 0) {
    spdlog::critical("mpv_initialize failed: {}", mpv_error_string(err));
    throw std::runtime_error("mpv_initialize failed");
  }
  mpv_request_log_messages(mpv_, "debug");

  mpv_observe_property(mpv_, 0, "time-pos", MPV_FORMAT_DOUBLE);
  mpv_observe_property(mpv_, 0, "duration", MPV_FORMAT_DOUBLE);
  mpv_observe_property(mpv_, 0, "pause", MPV_FORMAT_FLAG);
  mpv_set_wakeup_callback(mpv_, &MpvPlayer::HandleMpvWakeup, this);
}

bool MpvPlayer::openMedia(const std::string& source) {
  if (!mpv_ || source.empty())
    return false;
  const char* cmd[] = {"loadfile", source.c_str(), nullptr};
  return mpv_command(mpv_, cmd) >= 0;
}

bool MpvPlayer::play() {
  if (!mpv_)
    return false;
  const int status = mpv_set_property_string(mpv_, "pause", "no");
  return status >= 0;
}

void MpvPlayer::pause() {
  if (!mpv_)
    return;
  mpv_set_property_string(mpv_, "pause", "yes");
}

void MpvPlayer::stop() {
  if (!mpv_)
    return;
  mpv_command_string(mpv_, "stop");
}

void MpvPlayer::setVolume(double volume01) {
  if (!mpv_)
    return;
  double vol_percent = Clamp01(volume01) * kDefaultVolumePercent;
  mpv_set_property(mpv_, "volume", MPV_FORMAT_DOUBLE, &vol_percent);
}

void MpvPlayer::seek(double positionSeconds) {
  if (!mpv_)
    return;
  const std::string pos = std::to_string(positionSeconds);
  const char* cmd[] = {"seek", pos.c_str(), "absolute", nullptr};
  mpv_command(mpv_, cmd);
}

void MpvPlayer::setSharedMemorySink(std::shared_ptr<VideoSharedMemorySink> sink) {
  std::lock_guard<std::mutex> lock(sinkMutex_);
  sink_ = std::move(sink);
}

void MpvPlayer::setVideoShmMaxSize(std::uint32_t max_width, std::uint32_t max_height) {
  shm_max_width_.store(max_width, std::memory_order_relaxed);
  shm_max_height_.store(max_height, std::memory_order_relaxed);
}

void MpvPlayer::setVideoShmMaxFps(double max_fps) {
  if (max_fps < 0.0)
    max_fps = 0.0;
  shm_max_fps_.store(max_fps, std::memory_order_relaxed);
  shm_frame_interval_s_.store(max_fps > 0.0 ? (1.0 / max_fps) : 0.0, std::memory_order_relaxed);
}

std::uint32_t MpvPlayer::videoShmMaxWidth() const {
  return shm_max_width_.load(std::memory_order_relaxed);
}
std::uint32_t MpvPlayer::videoShmMaxHeight() const {
  return shm_max_height_.load(std::memory_order_relaxed);
}
double MpvPlayer::videoShmMaxFps() const {
  return shm_max_fps_.load(std::memory_order_relaxed);
}

bool MpvPlayer::initializeGl() {
  if (glInitialized_)
    return true;
  if (!mpv_)
    return false;
  if (!EnsureOpenGLFunctionsLoaded())
    return false;

  mpv_opengl_init_params init_params = {};
  init_params.get_proc_address = &MpvPlayer::GetProcAddress;
  init_params.get_proc_address_ctx = nullptr;

  mpv_render_param params[] = {
      {MPV_RENDER_PARAM_API_TYPE, const_cast<char*>(MPV_RENDER_API_TYPE_OPENGL)},
      {MPV_RENDER_PARAM_OPENGL_INIT_PARAMS, &init_params},
      {MPV_RENDER_PARAM_INVALID, nullptr},
  };

  if (mpv_render_context_create(&renderContext_, mpv_, params) < 0) {
    spdlog::error("mpv_render_context_create failed");
    renderContext_ = nullptr;
    return false;
  }

  mpv_render_context_set_update_callback(renderContext_, &MpvPlayer::HandleRenderUpdate, this);
  glInitialized_ = true;
  renderUpdatePending_.store(true, std::memory_order_release);
  return true;
}

void MpvPlayer::shutdownGl() {
  if (!glInitialized_)
    return;
  glInitialized_ = false;

  if (renderContext_) {
    mpv_render_context_set_update_callback(renderContext_, nullptr, nullptr);
    mpv_render_context_free(renderContext_);
    renderContext_ = nullptr;
  }

  destroyReadbackPbos();

  if (videoTextureId_)
    glDeleteTextures(1, &videoTextureId_);
  if (videoFboId_)
    glDeleteFramebuffers(1, &videoFboId_);
  videoTextureId_ = 0;
  videoFboId_ = 0;
  videoTexWidth_ = 0;
  videoTexHeight_ = 0;

  if (downsampleTextureId_)
    glDeleteTextures(1, &downsampleTextureId_);
  if (downsampleFboId_)
    glDeleteFramebuffers(1, &downsampleFboId_);
  downsampleTextureId_ = 0;
  downsampleFboId_ = 0;
  downsampleWidth_ = 0;
  downsampleHeight_ = 0;
}

bool MpvPlayer::renderVideoFrame() {
  if (!initializeGl())
    return false;
  if (!renderContext_ || !glInitialized_)
    return false;

  if (!renderUpdatePending_.exchange(false, std::memory_order_acq_rel))
    return false;
  uint64_t flags = mpv_render_context_update(renderContext_);
  if ((flags & MPV_RENDER_UPDATE_FRAME) == 0)
    return false;

  const unsigned width = videoWidth_.load(std::memory_order_relaxed);
  const unsigned height = videoHeight_.load(std::memory_order_relaxed);
  if (width == 0 || height == 0)
    return false;

  if (!ensureVideoTargets(width, height))
    return false;

  GLint prev_fbo = 0;
  GLint prev_viewport[4] = {};
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
  glGetIntegerv(GL_VIEWPORT, prev_viewport);

  glBindFramebuffer(GL_FRAMEBUFFER, videoFboId_);
  glViewport(0, 0, static_cast<GLint>(width), static_cast<GLint>(height));

  mpv_opengl_fbo fbo = {static_cast<int>(videoFboId_), static_cast<int>(width), static_cast<int>(height), 0};
  int flip = 1;
  mpv_render_param params[] = {
      {MPV_RENDER_PARAM_OPENGL_FBO, &fbo},
      {MPV_RENDER_PARAM_FLIP_Y, &flip},
      {MPV_RENDER_PARAM_INVALID, nullptr},
  };

  const int status = mpv_render_context_render(renderContext_, params);
  glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
  glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
  if (status < 0) {
    spdlog::warn("mpv render failed: {}", mpv_error_string(status));
    return false;
  }

  double shm_copy_ms = 0.0;
  copyFrameToSharedMemory(width, height, shm_copy_ms);
  return true;
}

void MpvPlayer::startEventThread() {
  stopRequested_.store(false, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(eventMutex_);
    eventNotified_ = true;
  }
  eventThread_ = std::thread(&MpvPlayer::eventLoop, this);
}

void MpvPlayer::stopEventThread() {
  stopRequested_.store(true, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(eventMutex_);
    eventNotified_ = true;
  }
  eventCv_.notify_all();
  if (mpv_)
    mpv_wakeup(mpv_);
  if (eventThread_.joinable())
    eventThread_.join();
}

void MpvPlayer::eventLoop() {
  while (!stopRequested_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lock(eventMutex_);
      eventCv_.wait(lock, [this] { return stopRequested_.load(std::memory_order_relaxed) || eventNotified_; });
      eventNotified_ = false;
    }
    if (!mpv_)
      break;
    while (!stopRequested_.load(std::memory_order_relaxed)) {
      mpv_event* event = mpv_wait_event(mpv_, 0);
      if (!event || event->event_id == MPV_EVENT_NONE)
        break;
      processEvent(*event);
      if (event->event_id == MPV_EVENT_SHUTDOWN) {
        stopRequested_.store(true, std::memory_order_relaxed);
        break;
      }
    }
  }
}

void MpvPlayer::processEvent(const mpv_event& event) {
  switch (event.event_id) {
    case MPV_EVENT_FILE_LOADED:
      if (playingCallback_)
        playingCallback_(true);
      break;
    case MPV_EVENT_LOG_MESSAGE:
      LogMpvMessage(static_cast<mpv_event_log_message*>(event.data));
      break;
    case MPV_EVENT_END_FILE:
      if (playingCallback_)
        playingCallback_(false);
      if (finishedCallback_) {
        int reason = 0;
        if (event.data) {
          const auto* end = static_cast<const mpv_event_end_file*>(event.data);
          reason = end ? end->reason : 0;
        }
        // Auto-advance playlist only on natural EOF; ignore stop/error.
        if (reason == MPV_END_FILE_REASON_EOF) {
          finishedCallback_();
        }
      }
      break;
    case MPV_EVENT_PROPERTY_CHANGE:
      if (event.data)
        handlePropertyChange(*static_cast<mpv_event_property*>(event.data));
      break;
    case MPV_EVENT_VIDEO_RECONFIG:
      handleVideoReconfig();
      break;
    default:
      break;
  }
}

void MpvPlayer::handlePropertyChange(const mpv_event_property& property) {
  if (!property.name)
    return;
  if (std::strcmp(property.name, "time-pos") == 0 && property.format == MPV_FORMAT_DOUBLE) {
    const double pos = property.data ? *static_cast<double*>(property.data) : 0.0;
    lastPositionSeconds_.store(pos, std::memory_order_relaxed);
    if (timeCallback_)
      timeCallback_(pos, lastDurationSeconds_.load(std::memory_order_relaxed));
  } else if (std::strcmp(property.name, "duration") == 0 && property.format == MPV_FORMAT_DOUBLE) {
    const double duration = property.data ? *static_cast<double*>(property.data) : 0.0;
    lastDurationSeconds_.store(duration, std::memory_order_relaxed);
    if (timeCallback_)
      timeCallback_(lastPositionSeconds_.load(std::memory_order_relaxed), duration);
  } else if (std::strcmp(property.name, "pause") == 0 && property.format == MPV_FORMAT_FLAG) {
    const int paused = property.data ? *static_cast<int*>(property.data) : 0;
    if (playingCallback_)
      playingCallback_(paused == 0);
  }
}

void MpvPlayer::handleVideoReconfig() {
  if (!mpv_)
    return;
  int64_t width = 0;
  int64_t height = 0;
  if (mpv_get_property(mpv_, "dwidth", MPV_FORMAT_INT64, &width) < 0 ||
      mpv_get_property(mpv_, "dheight", MPV_FORMAT_INT64, &height) < 0) {
    return;
  }
  if (width <= 0 || height <= 0)
    return;
  videoWidth_.store(static_cast<unsigned>(width), std::memory_order_relaxed);
  videoHeight_.store(static_cast<unsigned>(height), std::memory_order_relaxed);
}

std::pair<unsigned, unsigned> MpvPlayer::targetDimensions(unsigned width, unsigned height) const {
  if (width == 0 || height == 0)
    return {0, 0};
  double out_w = static_cast<double>(width);
  double out_h = static_cast<double>(height);
  double scale = 1.0;
  const auto max_w = shm_max_width_.load(std::memory_order_relaxed);
  const auto max_h = shm_max_height_.load(std::memory_order_relaxed);
  if (max_w > 0)
    scale = std::min(scale, static_cast<double>(max_w) / out_w);
  if (max_h > 0)
    scale = std::min(scale, static_cast<double>(max_h) / out_h);
  if (scale < 1.0) {
    out_w = std::max(1.0, std::floor(out_w * scale));
    out_h = std::max(1.0, std::floor(out_h * scale));
  }
  return {static_cast<unsigned>(out_w), static_cast<unsigned>(out_h)};
}

std::shared_ptr<VideoSharedMemorySink> MpvPlayer::sharedSink() const {
  std::lock_guard<std::mutex> lock(sinkMutex_);
  return sink_;
}

bool MpvPlayer::ensureVideoTargets(unsigned width, unsigned height) {
  if (videoTextureId_ != 0 && videoTexWidth_ == width && videoTexHeight_ == height)
    return true;

  if (videoTextureId_ == 0)
    glGenTextures(1, &videoTextureId_);
  glBindTexture(GL_TEXTURE_2D, videoTextureId_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLint>(width), static_cast<GLint>(height), 0, GL_BGRA,
               GL_UNSIGNED_BYTE, nullptr);

  if (videoFboId_ == 0)
    glGenFramebuffers(1, &videoFboId_);
  glBindFramebuffer(GL_FRAMEBUFFER, videoFboId_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, videoTextureId_, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    spdlog::error("video framebuffer incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return false;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  videoTexWidth_ = width;
  videoTexHeight_ = height;
  return true;
}

bool MpvPlayer::ensureDownsampleTargets(unsigned width, unsigned height) {
  if (downsampleTextureId_ != 0 && downsampleWidth_ == width && downsampleHeight_ == height)
    return true;

  if (downsampleTextureId_ == 0)
    glGenTextures(1, &downsampleTextureId_);
  glBindTexture(GL_TEXTURE_2D, downsampleTextureId_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLint>(width), static_cast<GLint>(height), 0, GL_BGRA,
               GL_UNSIGNED_BYTE, nullptr);

  if (downsampleFboId_ == 0)
    glGenFramebuffers(1, &downsampleFboId_);
  glBindFramebuffer(GL_FRAMEBUFFER, downsampleFboId_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, downsampleTextureId_, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    spdlog::error("downsample framebuffer incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return false;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  downsampleWidth_ = width;
  downsampleHeight_ = height;
  return true;
}

bool MpvPlayer::ensureReadbackPbos(std::size_t bytes) {
  if (bytes == 0)
    return false;
  if (readbackPbos_[0] == 0 && readbackPbos_[1] == 0) {
    glGenBuffers(kReadbackPboCount, readbackPbos_);
    readbackPboSize_ = 0;
    readbackPboIndex_ = 0;
    readbackPboInUse_[0] = readbackPboInUse_[1] = false;
  }
  if (readbackPboSize_ != bytes) {
    for (int i = 0; i < kReadbackPboCount; ++i) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, readbackPbos_[i]);
      glBufferData(GL_PIXEL_PACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    readbackPboSize_ = bytes;
    readbackPboInUse_[0] = readbackPboInUse_[1] = false;
    readbackPboIndex_ = 0;
  }
  return true;
}

void MpvPlayer::destroyReadbackPbos() {
  if (readbackPbos_[0] || readbackPbos_[1]) {
    glDeleteBuffers(kReadbackPboCount, readbackPbos_);
    readbackPbos_[0] = readbackPbos_[1] = 0;
  }
  readbackPboSize_ = 0;
  readbackPboIndex_ = 0;
  readbackPboInUse_[0] = readbackPboInUse_[1] = false;
}

bool MpvPlayer::shouldCopyToSharedMemory(std::chrono::steady_clock::time_point now) const {
  if (config_.offline)
    return false;
  const double interval_s = shm_frame_interval_s_.load(std::memory_order_relaxed);
  if (interval_s <= 0.0)
    return true;
  if (lastShmFrameTime_.time_since_epoch().count() == 0)
    return true;
  const double elapsed_s = std::chrono::duration<double>(now - lastShmFrameTime_).count();
  return elapsed_s >= interval_s;
}

bool MpvPlayer::copyFrameToSharedMemory(unsigned width, unsigned height, double&) {
  auto sink = sharedSink();
  if (!sink)
    return false;
  const auto now = std::chrono::steady_clock::now();
  if (!shouldCopyToSharedMemory(now))
    return false;

  auto [target_w, target_h] = targetDimensions(width, height);
  if (target_w == 0 || target_h == 0)
    return false;
  if (!sink->ensureConfiguration(target_w, target_h))
    return false;
  if (!ensureDownsampleTargets(target_w, target_h))
    return false;

  GLint prev_read = 0;
  GLint prev_draw = 0;
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prev_read);
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev_draw);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, videoFboId_);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, downsampleFboId_);
  // glReadPixels reads from bottom-left; flip during blit so the SHM payload is top-down.
  glBlitFramebuffer(0, 0, static_cast<GLint>(width), static_cast<GLint>(height), 0, static_cast<GLint>(target_h),
                    static_cast<GLint>(target_w), 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);

  glBindFramebuffer(GL_FRAMEBUFFER, downsampleFboId_);

  const std::size_t byte_count = static_cast<std::size_t>(target_w) * target_h * 4;
  if (!ensureReadbackPbos(byte_count)) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_draw);
    return false;
  }

  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  const int current_index = readbackPboIndex_;
  glBindBuffer(GL_PIXEL_PACK_BUFFER, readbackPbos_[current_index]);
  glBufferData(GL_PIXEL_PACK_BUFFER, static_cast<GLsizeiptr>(byte_count), nullptr, GL_STREAM_READ);
  glReadPixels(0, 0, static_cast<GLint>(target_w), static_cast<GLint>(target_h), GL_BGRA, GL_UNSIGNED_BYTE,
               reinterpret_cast<void*>(0));
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_draw);

  readbackPboInUse_[current_index] = true;

  bool wrote_frame = false;
  const int ready_index = (current_index + kReadbackPboCount - 1) % kReadbackPboCount;
  if (readbackPboInUse_[ready_index]) {
    glBindBuffer(GL_PIXEL_PACK_BUFFER, readbackPbos_[ready_index]);
    void* mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr>(byte_count), GL_MAP_READ_BIT);
    if (mapped) {
      sink->writeFrame(mapped, target_w * 4);
      wrote_frame = true;
      glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    readbackPboInUse_[ready_index] = false;
  }

  readbackPboIndex_ = (readbackPboIndex_ + 1) % kReadbackPboCount;
  if (wrote_frame)
    lastShmFrameTime_ = now;
  return wrote_frame;
}

void MpvPlayer::HandleMpvWakeup(void* userdata) {
  auto* self = static_cast<MpvPlayer*>(userdata);
  if (!self)
    return;
  {
    std::lock_guard<std::mutex> lock(self->eventMutex_);
    self->eventNotified_ = true;
  }
  self->eventCv_.notify_one();
}

void MpvPlayer::HandleRenderUpdate(void* userdata) {
  auto* self = static_cast<MpvPlayer*>(userdata);
  if (!self)
    return;
  self->renderUpdatePending_.store(true, std::memory_order_release);
}

void* MpvPlayer::GetProcAddress(void*, const char* name) {
  return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name));
}

}  // namespace f8::implayer
