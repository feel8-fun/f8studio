#include "audiocap_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <SDL3/SDL.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/shm/audio.h"
#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/time_utils.h"
#include "wasapi_loopback_capture.h"

namespace f8::audiocap {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_number() { return json{{"type", "number"}}; }
json schema_integer() { return json{{"type", "integer"}}; }
json schema_boolean() { return json{{"type", "boolean"}}; }
json schema_object(const json& props, const json& required = json::array()) {
  json obj;
  obj["type"] = "object";
  obj["properties"] = props;
  if (required.is_array()) obj["required"] = required;
  obj["additionalProperties"] = false;
  return obj;
}

json state_field(std::string name, const json& value_schema, std::string access, std::string label = {},
                 std::string description = {}, bool show_on_node = false) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  if (!label.empty()) sf["label"] = std::move(label);
  if (!description.empty()) sf["description"] = std::move(description);
  if (show_on_node) sf["showOnNode"] = true;
  return sf;
}

double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

}  // namespace

AudioCapService::AudioCapService(Config cfg) : cfg_(std::move(cfg)) {}

AudioCapService::~AudioCapService() { stop(); }

std::string AudioCapService::default_audio_shm_name(const std::string& service_id) {
  return f8::cppsdk::shm::audio_shm_name(service_id);
}

namespace {

bool is_digits(const std::string& s) {
  if (s.empty()) return false;
  for (unsigned char ch : s) {
    if (ch < '0' || ch > '9') return false;
  }
  return true;
}

bool contains_icase(std::string_view haystack, std::string_view needle) {
  if (needle.empty()) return true;
  if (haystack.empty()) return false;
  std::string hs(haystack);
  std::string nd(needle);
  std::transform(hs.begin(), hs.end(), hs.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  std::transform(nd.begin(), nd.end(), nd.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return hs.find(nd) != std::string::npos;
}

SDL_AudioDeviceID pick_recording_device(const std::string& selector, std::string& out_name) {
  out_name.clear();
  int count = 0;
  SDL_AudioDeviceID* devices = SDL_GetAudioRecordingDevices(&count);
  if (!devices || count <= 0) {
    if (devices) SDL_free(devices);
    return SDL_AUDIO_DEVICE_DEFAULT_RECORDING;
  }

  auto finish = [&](SDL_AudioDeviceID id) {
    if (devices) SDL_free(devices);
    return id;
  };

  if (selector.empty()) {
    return finish(SDL_AUDIO_DEVICE_DEFAULT_RECORDING);
  }

  if (is_digits(selector)) {
    const int idx = std::stoi(selector);
    if (idx >= 0 && idx < count) {
      const SDL_AudioDeviceID id = devices[idx];
      const char* nm = SDL_GetAudioDeviceName(id);
      if (nm) out_name = nm;
      return finish(id);
    }
  }

  for (int i = 0; i < count; ++i) {
    const SDL_AudioDeviceID id = devices[i];
    const char* nm = SDL_GetAudioDeviceName(id);
    if (!nm) continue;
    if (contains_icase(nm, selector)) {
      out_name = nm;
      return finish(id);
    }
  }

  return finish(SDL_AUDIO_DEVICE_DEFAULT_RECORDING);
}

}  // namespace

bool AudioCapService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  try {
    cfg_.service_id = f8::cppsdk::ensure_token(cfg_.service_id, "service_id");
  } catch (const std::exception& e) {
    spdlog::error("invalid --service-id: {}", e.what());
    return false;
  } catch (...) {
    spdlog::error("invalid --service-id");
    return false;
  }

  cfg_.gain = clamp01(cfg_.gain);
  if (cfg_.frames_per_chunk == 0) cfg_.frames_per_chunk = 480;
  if (cfg_.chunk_count == 0) cfg_.chunk_count = 200;

  if (!SDL_Init(SDL_INIT_AUDIO)) {
    spdlog::error("SDL_Init(SDL_INIT_AUDIO) failed: {}", SDL_GetError());
    return false;
  }

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_set_state_node(this);
  bus_->add_rungraph_node(this);
  bus_->add_command_node(this);
  if (!bus_->start()) return false;

  shm_ = std::make_unique<f8::cppsdk::AudioSharedMemorySink>();
  const std::string shm_name = f8::cppsdk::shm::audio_shm_name(cfg_.service_id);
  if (!shm_->initialize(shm_name, cfg_.audio_shm_bytes, cfg_.sample_rate, cfg_.channels,
                        f8::cppsdk::AudioSharedMemorySink::SampleFormat::kF32LE, cfg_.frames_per_chunk,
                        cfg_.chunk_count)) {
    spdlog::error("failed to initialize audio shm sink name={} bytes={}", shm_name, cfg_.audio_shm_bytes);
    return false;
  }

  chunk_buffer_.assign(static_cast<std::size_t>(cfg_.frames_per_chunk) * cfg_.channels, 0.0f);
  capture_chunk_accum_.assign(static_cast<std::size_t>(cfg_.frames_per_chunk) * cfg_.channels, 0.0f);
  capture_accum_frames_ = 0;
  phase_ = 0.0;
  last_write_ms_ = 0;
  last_state_pub_ms_ = 0;

  opened_device_name_.clear();
  opened_device_ = 0;
  stream_ = nullptr;
  wasapi_.reset();

  if (cfg_.mode == "capture") {
#if defined(_WIN32)
    const bool want_wasapi = (cfg_.backend.empty() || cfg_.backend == "auto" || cfg_.backend == "wasapi");
    if (want_wasapi) {
      wasapi_ = std::make_unique<WasapiLoopbackCapture>(
          WasapiLoopbackCapture::Config{static_cast<std::uint32_t>(cfg_.sample_rate), cfg_.channels});
      std::string err;
      std::string dev;
      const bool ok = wasapi_->start(
          [this](const float* interleaved, std::uint32_t frames, std::int64_t ts_ms) {
            this->handle_captured_interleaved_f32(interleaved, frames, ts_ms);
          },
          dev, err);
      if (!ok) {
        spdlog::warn("WASAPI loopback start failed: {} (falling back to SDL recording)", err);
        wasapi_.reset();
      } else {
        opened_device_name_ = "WASAPI(loopback): " + dev;
      }
    }
#endif

    if (wasapi_) {
      publish_static_state();
      publish_dynamic_state();
      running_.store(true, std::memory_order_release);
      stop_requested_.store(false, std::memory_order_release);
      spdlog::info("audiocap started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
      return true;
    }

    std::string matched;
    const SDL_AudioDeviceID devid = pick_recording_device(cfg_.device, matched);

    SDL_AudioSpec spec;
    spec.format = SDL_AUDIO_F32;
    spec.channels = static_cast<int>(cfg_.channels);
    spec.freq = static_cast<int>(cfg_.sample_rate);

    stream_ = SDL_OpenAudioDeviceStream(devid, &spec, nullptr, nullptr);
    if (!stream_) {
      spdlog::error("SDL_OpenAudioDeviceStream failed: {}", SDL_GetError());
      return false;
    }

    opened_device_ = SDL_GetAudioStreamDevice(stream_);
    const char* nm = SDL_GetAudioDeviceName(opened_device_);
    opened_device_name_ = nm ? nm : matched;

    SDL_SetAudioStreamPutCallback(stream_, &AudioCapService::on_audio_stream_put, this);
    if (!SDL_ResumeAudioStreamDevice(stream_)) {
      spdlog::error("SDL_ResumeAudioStreamDevice failed: {}", SDL_GetError());
      return false;
    }
  }

  publish_static_state();
  publish_dynamic_state();

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("audiocap started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void AudioCapService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  stop_requested_.store(true, std::memory_order_release);

  if (stream_) {
    SDL_DestroyAudioStream(stream_);
    stream_ = nullptr;
  }
  if (wasapi_) {
    wasapi_->stop();
    wasapi_.reset();
  }
  opened_device_ = 0;
  opened_device_name_.clear();

  shm_.reset();
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();

  SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

void AudioCapService::tick() {
  if (!running_.load(std::memory_order_acquire)) return;

  if (bus_) {
    (void)bus_->drain_main_thread();
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (now - last_state_pub_ms_ >= 200) {
    publish_dynamic_state();
    last_state_pub_ms_ = now;
  }

  if (!active_.load(std::memory_order_acquire)) return;
  if (!shm_) return;

  if (cfg_.mode == "capture") {
    return;
  }

  const std::int64_t chunk_ms = static_cast<std::int64_t>(
      std::llround(1000.0 * static_cast<double>(cfg_.frames_per_chunk) / static_cast<double>(cfg_.sample_rate)));
  if (chunk_ms > 0 && last_write_ms_ != 0 && (now - last_write_ms_) < chunk_ms) {
    return;
  }
  last_write_ms_ = now;

  if (cfg_.mode == "sine") {
    const double dt = 1.0 / static_cast<double>(cfg_.sample_rate);
    const double w = 2.0 * 3.14159265358979323846 * cfg_.tone_hz;
    for (std::uint32_t i = 0; i < cfg_.frames_per_chunk; ++i) {
      const float s = static_cast<float>(std::sin(phase_) * cfg_.gain);
      phase_ += w * dt;
      if (phase_ > 2.0 * 3.14159265358979323846) phase_ -= 2.0 * 3.14159265358979323846;
      for (std::uint16_t c = 0; c < cfg_.channels; ++c) {
        chunk_buffer_[static_cast<std::size_t>(i) * cfg_.channels + c] = s;
      }
    }
  } else {
    std::fill(chunk_buffer_.begin(), chunk_buffer_.end(), 0.0f);
  }

  (void)shm_->write_interleaved_f32(chunk_buffer_.data(), cfg_.frames_per_chunk, now);
}

void SDLCALL AudioCapService::on_audio_stream_put(void* userdata, SDL_AudioStream* stream, int additional_amount,
                                                 int total_amount) {
  auto* self = static_cast<AudioCapService*>(userdata);
  if (!self) return;
  self->handle_audio_stream_put(stream, additional_amount, total_amount);
}

void AudioCapService::handle_audio_stream_put(SDL_AudioStream* stream, int additional_amount, int total_amount) {
  (void)total_amount;
  if (!running_.load(std::memory_order_acquire)) return;
  if (!active_.load(std::memory_order_acquire)) {
    // Drain and drop.
    int avail = SDL_GetAudioStreamAvailable(stream);
    if (avail > 0) {
      const int frame_bytes = static_cast<int>(sizeof(float) * cfg_.channels);
      const int clamped = (avail / frame_bytes) * frame_bytes;
      if (clamped > 0) {
        capture_tmp_.resize(static_cast<std::size_t>(clamped / static_cast<int>(sizeof(float))));
        (void)SDL_GetAudioStreamData(stream, capture_tmp_.data(), clamped);
      }
    }
    return;
  }
  if (!shm_) return;

  const int frame_bytes = static_cast<int>(sizeof(float) * cfg_.channels);
  int avail = SDL_GetAudioStreamAvailable(stream);
  if (avail < frame_bytes) return;

  // Read in reasonable chunks to avoid large allocations.
  int want = additional_amount > 0 ? additional_amount : avail;
  want = std::min(want, avail);
  want = std::max(want, frame_bytes);
  want = (want / frame_bytes) * frame_bytes;
  if (want <= 0) return;

  capture_tmp_.resize(static_cast<std::size_t>(want / static_cast<int>(sizeof(float))));
  const int got = SDL_GetAudioStreamData(stream, capture_tmp_.data(), want);
  if (got <= 0) return;

  const int got_aligned = (got / frame_bytes) * frame_bytes;
  const std::uint32_t frames = static_cast<std::uint32_t>(got_aligned / frame_bytes);

  const std::int64_t ts_ms = f8::cppsdk::now_ms();
  handle_captured_interleaved_f32(capture_tmp_.data(), frames, ts_ms);
}

void AudioCapService::handle_captured_interleaved_f32(const float* interleaved, std::uint32_t frames,
                                                      std::int64_t ts_ms) {
  if (!interleaved || frames == 0) return;
  if (!running_.load(std::memory_order_acquire)) return;
  if (!active_.load(std::memory_order_acquire)) return;
  if (!shm_) return;

  const float* src = interleaved;
  std::uint32_t frames_left = frames;

  while (frames_left > 0) {
    const std::uint32_t room = cfg_.frames_per_chunk - capture_accum_frames_;
    const std::uint32_t take = std::min(room, frames_left);

    const std::size_t dst_off = static_cast<std::size_t>(capture_accum_frames_) * cfg_.channels;
    const std::size_t src_off = static_cast<std::size_t>(frames - frames_left) * cfg_.channels;
    std::memcpy(capture_chunk_accum_.data() + dst_off, src + src_off, take * cfg_.channels * sizeof(float));

    capture_accum_frames_ += take;
    frames_left -= take;

    if (capture_accum_frames_ == cfg_.frames_per_chunk) {
      (void)shm_->write_interleaved_f32(capture_chunk_accum_.data(), cfg_.frames_per_chunk, ts_ms);
      capture_accum_frames_ = 0;
    }
  }
}

void AudioCapService::set_active_local(bool active, const nlohmann::json& meta) {
  active_.store(active, std::memory_order_release);
  if (cfg_.mode == "capture" && stream_) {
    if (active) {
      (void)SDL_ResumeAudioStreamDevice(stream_);
    } else {
      (void)SDL_PauseAudioStreamDevice(stream_);
    }
  }
  if (cfg_.mode == "capture" && wasapi_) {
    wasapi_->set_paused(!active);
  }
  (void)meta;
}

void AudioCapService::on_lifecycle(bool active, const nlohmann::json& meta) { set_active_local(active, meta); }

void AudioCapService::on_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                               std::int64_t ts_ms, const nlohmann::json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;
  std::string ec;
  std::string em;
  json result;
  (void)on_set_state(node_id, field, value, meta, ec, em);
}

bool AudioCapService::on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                                   const nlohmann::json& meta, std::string& error_code, std::string& error_message) {
  (void)node_id;
  (void)field;
  (void)value;
  (void)meta;
  error_code = "not_supported";
  error_message = "field not supported";
  return false;
}

bool AudioCapService::on_set_rungraph(const nlohmann::json&, const nlohmann::json&, std::string& error_code,
                                      std::string& error_message) {
  error_code.clear();
  error_message.clear();
  return true;
}

bool AudioCapService::on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                                 nlohmann::json& result, std::string& error_code, std::string& error_message) {
  (void)args;
  (void)meta;
  (void)result;
  error_code = "not_supported";
  error_message = "unknown command: " + call;
  return false;
}

void AudioCapService::publish_static_state() {
  std::lock_guard<std::mutex> lock(state_mu_);

  auto set_if_changed = [&](const char* field, const nlohmann::json& v) {
    auto it = published_state_.find(field);
    if (it != published_state_.end() && it->second == v) return;
    published_state_[field] = v;
    if (bus_) {
      f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, v, "init", json::object());
    }
  };

  set_if_changed("serviceClass", cfg_.service_class);
  set_if_changed("audioShmName", shm_ ? shm_->shm_name() : "");
  set_if_changed("audioDevice", opened_device_name_);
  set_if_changed("audioSampleRate", cfg_.sample_rate);
  set_if_changed("audioChannels", cfg_.channels);
  set_if_changed("audioFormat", "f32le");
  set_if_changed("audioFramesPerChunk", cfg_.frames_per_chunk);
  set_if_changed("audioChunkCount", cfg_.chunk_count);
  set_if_changed("mode", cfg_.mode);
  set_if_changed("toneHz", cfg_.tone_hz);
  set_if_changed("gain", cfg_.gain);
}

void AudioCapService::publish_dynamic_state() {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto set_if_changed = [&](const char* field, const nlohmann::json& v) {
    auto it = published_state_.find(field);
    if (it != published_state_.end() && it->second == v) return;
    published_state_[field] = v;
    if (bus_) {
      f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, v, "tick", json::object());
    }
  };
  if (shm_) set_if_changed("writeSeq", static_cast<std::uint64_t>(shm_->write_seq()));
}

nlohmann::json AudioCapService::describe() {
  json spec;
  spec["service"] = {
      {"schemaVersion", "f8service/1"},
      {"serviceClass", "f8.audiocap"},
      {"label", "Audio Capture"},
      {"version", "0.0.1"},
      {"rendererClass", "default_svc"},
      {"tags", json::array({"audio", "capture", "shm"})},
      {"stateFields",
       json::array({
           state_field("audioShmName", schema_string(), "ro", "Audio SHM", "Name of the audio shared memory segment", true),
           state_field("audioDevice", schema_string(), "ro", "Audio Device", "Name of the audio capture device in use", false),
           state_field("audioSampleRate", schema_integer(), "ro", "Audio Sample Rate", "Sample rate of the audio capture device", false),
           state_field("audioChannels", schema_integer(), "ro", "Audio Channels", "Number of audio channels", false),
           state_field("audioFormat", schema_string(), "ro", "Audio Format", "Format of the audio data", false),
           state_field("audioFramesPerChunk", schema_integer(), "ro", "Audio Frames Per Chunk", "Number of audio frames per chunk", false),
           state_field("audioChunkCount", schema_integer(), "ro", "Audio Chunk Count", "Number of audio chunks", false),
           state_field("writeSeq", schema_integer(), "ro", "Write Sequence", "Sequence number of the last written audio chunk", false),
           state_field("mode", schema_string(), "rw", "Mode", "Current mode of the audio capture service", false),
           state_field("toneHz", schema_number(), "rw", "Tone Frequency", "Frequency of the generated tone", false),
           state_field("gain", schema_number(), "rw", "Gain", "Gain applied to the audio signal", false),
       })},
      {"editableStateFields", false},
      {"commands", json::array()},
      {"editableCommands", false},
      {"dataInPorts", json::array()},
      {"dataOutPorts", json::array()},
      {"editableDataInPorts", false},
      {"editableDataOutPorts", false},
  };
  spec["operators"] = json::array();
  return spec;
}

}  // namespace f8::audiocap
