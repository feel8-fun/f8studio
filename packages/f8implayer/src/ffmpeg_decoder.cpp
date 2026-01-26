#include "ffmpeg_decoder.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include <spdlog/spdlog.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include "f8cppsdk/video_shared_memory_sink.h"

namespace f8::implayer {

namespace {

double to_seconds(int64_t pts, AVRational time_base) {
  if (pts == AV_NOPTS_VALUE)
    return 0.0;
  return static_cast<double>(pts) * av_q2d(time_base);
}

std::int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

FfmpegDecoder::FfmpegDecoder(Config cfg, std::shared_ptr<VideoSharedMemorySink> sink)
    : cfg_(cfg), sink_(std::move(sink)) {
  thread_ = std::thread(&FfmpegDecoder::worker, this);
}

FfmpegDecoder::~FfmpegDecoder() {
  stop_.store(true, std::memory_order_release);
  playing_.store(false, std::memory_order_release);
  if (thread_.joinable())
    thread_.join();
}

std::string FfmpegDecoder::url() const {
  std::lock_guard<std::mutex> lock(mu_);
  return url_;
}

std::string FfmpegDecoder::last_error() const {
  std::lock_guard<std::mutex> lock(mu_);
  return last_error_;
}

void FfmpegDecoder::set_error(std::string err) {
  std::lock_guard<std::mutex> lock(mu_);
  last_error_ = std::move(err);
}

bool FfmpegDecoder::open(const std::string& url, std::string& err) {
  if (url.empty()) {
    err = "url is empty";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    url_ = url;
    last_error_.clear();
  }
  reopen_.store(true, std::memory_order_release);
  playing_.store(true, std::memory_order_release);
  return true;
}

void FfmpegDecoder::close() {
  playing_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mu_);
    url_.clear();
  }
  reopen_.store(true, std::memory_order_release);
}

void FfmpegDecoder::play() {
  playing_.store(true, std::memory_order_release);
}
void FfmpegDecoder::pause() {
  playing_.store(false, std::memory_order_release);
}
void FfmpegDecoder::stop() {
  close();
}

bool FfmpegDecoder::seek(double position_seconds, std::string& err) {
  if (position_seconds < 0.0)
    position_seconds = 0.0;
  if (url().empty()) {
    err = "no media open";
    return false;
  }
  seek_req_s_.store(position_seconds, std::memory_order_release);
  return true;
}

void FfmpegDecoder::worker() {
  AVFormatContext* fmt = nullptr;
  AVCodecContext* dec = nullptr;
  SwsContext* sws = nullptr;
  AVPacket* pkt = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  AVFrame* bgra = av_frame_alloc();
  std::vector<std::uint8_t> bgra_buf;

  int video_stream = -1;
  AVRational tb = {1, 1};
  double max_fps = cfg_.max_fps;
  const bool throttle = max_fps > 0.0;
  const auto min_interval_ms = throttle ? static_cast<std::int64_t>(1000.0 / max_fps) : 0;
  std::int64_t last_push_ms = 0;

  auto cleanup = [&]() {
    if (sws) {
      sws_freeContext(sws);
      sws = nullptr;
    }
    if (dec) {
      avcodec_free_context(&dec);
      dec = nullptr;
    }
    if (fmt) {
      avformat_close_input(&fmt);
      fmt = nullptr;
    }
    video_stream = -1;
    tb = {1, 1};
  };

  auto reopen = [&]() -> bool {
    cleanup();
    std::string url_s = url();
    if (url_s.empty()) {
      return false;
    }

    fmt = avformat_alloc_context();
    if (!fmt) {
      set_error("avformat_alloc_context failed");
      return false;
    }

    if (avformat_open_input(&fmt, url_s.c_str(), nullptr, nullptr) != 0) {
      set_error("avformat_open_input failed");
      cleanup();
      return false;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
      set_error("avformat_find_stream_info failed");
      cleanup();
      return false;
    }

    video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream < 0) {
      set_error("no video stream");
      cleanup();
      return false;
    }

    AVStream* st = fmt->streams[video_stream];
    tb = st->time_base;
    if (st->duration > 0 && st->duration != AV_NOPTS_VALUE) {
      duration_s_.store(static_cast<double>(st->duration) * av_q2d(tb), std::memory_order_release);
    } else if (fmt->duration > 0) {
      duration_s_.store(static_cast<double>(fmt->duration) / AV_TIME_BASE, std::memory_order_release);
    }

    const AVCodecParameters* par = st->codecpar;
    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
      set_error("avcodec_find_decoder failed");
      cleanup();
      return false;
    }
    dec = avcodec_alloc_context3(codec);
    if (!dec) {
      set_error("avcodec_alloc_context3 failed");
      cleanup();
      return false;
    }
    if (avcodec_parameters_to_context(dec, par) < 0) {
      set_error("avcodec_parameters_to_context failed");
      cleanup();
      return false;
    }
    if (avcodec_open2(dec, codec, nullptr) < 0) {
      set_error("avcodec_open2 failed");
      cleanup();
      return false;
    }

    return true;
  };

  while (!stop_.load(std::memory_order_acquire)) {
    if (reopen_.exchange(false, std::memory_order_acq_rel)) {
      if (!reopen()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      }
      last_push_ms = 0;
      position_s_.store(0.0, std::memory_order_release);
    }

    if (!fmt || !dec || video_stream < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    if (!playing_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    const double seek_s = seek_req_s_.exchange(-1.0, std::memory_order_acq_rel);
    if (seek_s >= 0.0) {
      const int64_t ts = static_cast<int64_t>(seek_s / av_q2d(tb));
      if (av_seek_frame(fmt, video_stream, ts, AVSEEK_FLAG_BACKWARD) < 0) {
        set_error("av_seek_frame failed");
      } else {
        avcodec_flush_buffers(dec);
        position_s_.store(seek_s, std::memory_order_release);
      }
    }

    if (av_read_frame(fmt, pkt) < 0) {
      playing_.store(false, std::memory_order_release);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    if (pkt->stream_index != video_stream) {
      av_packet_unref(pkt);
      continue;
    }

    if (avcodec_send_packet(dec, pkt) < 0) {
      av_packet_unref(pkt);
      continue;
    }
    av_packet_unref(pkt);

    while (!stop_.load(std::memory_order_acquire)) {
      int r = avcodec_receive_frame(dec, frame);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
        break;
      }
      if (r < 0) {
        set_error("avcodec_receive_frame failed");
        break;
      }

      const unsigned w = static_cast<unsigned>(frame->width);
      const unsigned h = static_cast<unsigned>(frame->height);
      if (w == 0 || h == 0) {
        av_frame_unref(frame);
        continue;
      }

      if (sink_) {
        sink_->ensureConfiguration(w, h);
      }

      sws = sws_getCachedContext(sws, frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                                 frame->width, frame->height, AV_PIX_FMT_BGRA, SWS_BILINEAR, nullptr, nullptr, nullptr);
      if (!sws) {
        set_error("sws_getCachedContext failed");
        av_frame_unref(frame);
        continue;
      }

      const int dst_stride = frame->width * 4;
      const std::size_t need = static_cast<std::size_t>(dst_stride) * frame->height;
      if (bgra_buf.size() != need) {
        bgra_buf.resize(need);
      }

      uint8_t* dst_data[4] = {bgra_buf.data(), nullptr, nullptr, nullptr};
      int dst_linesize[4] = {dst_stride, 0, 0, 0};
      sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);

      if (sink_) {
        sink_->writeFrame(bgra_buf.data(), static_cast<unsigned>(dst_stride));
      }

      const double pos = to_seconds(frame->best_effort_timestamp, tb);
      position_s_.store(pos, std::memory_order_release);

      av_frame_unref(frame);

      if (throttle) {
        const std::int64_t now = now_ms();
        if (last_push_ms > 0) {
          const auto elapsed = now - last_push_ms;
          if (elapsed < min_interval_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(min_interval_ms - elapsed));
          }
        }
        last_push_ms = now_ms();
      }
    }
  }

  cleanup();
  if (pkt)
    av_packet_free(&pkt);
  if (frame)
    av_frame_free(&frame);
  if (bgra)
    av_frame_free(&bgra);
}

}  // namespace f8::implayer
