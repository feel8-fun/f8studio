#include "audiocap_service.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include <SDL3/SDL.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/f8_naming.h"
#include "f8cppsdk/latest_audio_chunk_transport.h"
#include "f8cppsdk/time_utils.h"
#include "f8cppsdk/zenoh_naming.h"
#include "wasapi_loopback_capture.h"

namespace f8::audiocap {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_boolean;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_object;
using f8::cppsdk::describe::schema_string;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::audio_chunk_port;

namespace {

double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

}  // namespace

AudioCapService::AudioCapService(Config cfg) : cfg_(std::move(cfg)) {}

AudioCapService::~AudioCapService() { stop(); }

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

SDL_AudioDeviceID pick_recording_device(const std::string& selector, std::string& out_name, std::string& error) {
  out_name.clear();
  error.clear();
  const bool named_recording = selector.rfind("Recording: ", 0) == 0;
  const std::string name = named_recording ? selector.substr(11) : selector;
  if (name.empty() || name == "Default input") return SDL_AUDIO_DEVICE_DEFAULT_RECORDING;
  int count = 0;
  SDL_AudioDeviceID* devices = SDL_GetAudioRecordingDevices(&count);
  if (!devices || count <= 0) {
    if (devices) SDL_free(devices);
    error = "no recording devices are available";
    return 0;
  }

  auto finish = [&](SDL_AudioDeviceID id) {
    if (devices) SDL_free(devices);
    return id;
  };

  if (!named_recording && is_digits(selector)) {
    int idx = -1;
    const auto parsed = std::from_chars(selector.data(), selector.data() + selector.size(), idx);
    if (parsed.ec != std::errc{} || parsed.ptr != selector.data() + selector.size()) {
      error = "invalid recording device index: " + selector;
      return finish(0);
    }
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
    if ((named_recording && name == nm) || (!named_recording && contains_icase(nm, name))) {
      out_name = nm;
      return finish(id);
    }
  }

  error = "recording device not found: " + name;
  return finish(0);
}

}  // namespace

std::vector<std::string> AudioCapService::available_capture_devices() const {
  std::vector<std::string> devices{"Auto"};
#if defined(_WIN32)
  const auto render_devices = WasapiLoopbackCapture::available_render_devices();
  if (!render_devices.empty()) devices.push_back("Loopback: Default output");
  for (const std::string& name : render_devices) {
    const std::string option = "Loopback: " + name;
    if (std::find(devices.begin(), devices.end(), option) == devices.end()) devices.push_back(option);
  }
#endif
  int count = 0;
  SDL_AudioDeviceID* recording = SDL_GetAudioRecordingDevices(&count);
  if (recording && count > 0) {
    devices.push_back("Recording: Default input");
    for (int index = 0; index < count; ++index) {
      const char* name = SDL_GetAudioDeviceName(recording[index]);
      if (!name || !*name) continue;
      const std::string option = std::string("Recording: ") + name;
      if (std::find(devices.begin(), devices.end(), option) == devices.end()) devices.push_back(option);
    }
  }
  if (recording) SDL_free(recording);
  return devices;
}

void AudioCapService::close_capture_device() {
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
  capture_accum_frames_ = 0;
}

bool AudioCapService::open_capture_device(const std::string& selector, std::string& error) {
  error.clear();
  if (cfg_.mode != "capture") {
    selected_device_ = selector;
    return true;
  }

  const bool automatic = selector == "Auto";
  const bool loopback = selector.rfind("Loopback: ", 0) == 0;
  const bool recording = selector.rfind("Recording: ", 0) == 0;
  if (!automatic && !loopback && !recording && selector != cfg_.device) {
    error = "unknown capture device selection: " + selector;
    return false;
  }

#if defined(_WIN32)
  const bool use_wasapi = loopback || (automatic && cfg_.backend != "sdl");
  if (use_wasapi) {
    const std::string render_name = loopback && selector != "Loopback: Default output" ? selector.substr(10) : "";
    auto candidate = std::make_unique<WasapiLoopbackCapture>(
        WasapiLoopbackCapture::Config{cfg_.sample_rate, cfg_.channels, render_name});
    std::string device_name;
    if (candidate->start(
            [this](const float* interleaved, std::uint32_t frames, std::int64_t ts_ms) {
              handle_captured_interleaved_f32(interleaved, frames, ts_ms);
            },
            device_name, error)) {
      opened_device_name_ = "WASAPI(loopback): " + device_name;
      wasapi_ = std::move(candidate);
      selected_device_ = selector;
      wasapi_->set_paused(!active_.load(std::memory_order_acquire));
      return true;
    }
    if (!automatic || cfg_.backend == "wasapi") return false;
    spdlog::warn("WASAPI loopback unavailable, trying default recording device: {}", error);
  }
#else
  if (loopback) {
    error = "loopback capture is only available on Windows";
    return false;
  }
#endif

  const std::string recording_selector = automatic ? "" : selector;
  std::string matched;
  const SDL_AudioDeviceID device = pick_recording_device(recording_selector, matched, error);
  if (device == 0) return false;
  SDL_AudioSpec spec{};
  spec.format = SDL_AUDIO_F32;
  spec.channels = static_cast<int>(cfg_.channels);
  spec.freq = static_cast<int>(cfg_.sample_rate);
  SDL_AudioStream* opened = SDL_OpenAudioDeviceStream(device, &spec, nullptr, nullptr);
  if (!opened) {
    error = std::string("SDL_OpenAudioDeviceStream failed: ") + SDL_GetError();
    return false;
  }
  SDL_SetAudioStreamPutCallback(opened, &AudioCapService::on_audio_stream_put, this);
  if (!SDL_ResumeAudioStreamDevice(opened)) {
    error = std::string("SDL_ResumeAudioStreamDevice failed: ") + SDL_GetError();
    SDL_DestroyAudioStream(opened);
    return false;
  }
  stream_ = opened;
  opened_device_ = SDL_GetAudioStreamDevice(opened);
  const char* actual_name = SDL_GetAudioDeviceName(opened_device_);
  opened_device_name_ = actual_name ? actual_name : matched;
  selected_device_ = selector;
  if (!active_.load(std::memory_order_acquire)) (void)SDL_PauseAudioStreamDevice(stream_);
  return true;
}

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
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  bus_cfg.apply_runtime_backend(runtime_backend);
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "Audio Capture";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_set_state_node(this);
  bus_->add_rungraph_node(this);
  bus_->add_command_node(this, AudioCapService::describe());
  if (!bus_->start()) return false;

  zenoh_audio_seq_.store(0, std::memory_order_relaxed);
  zenoh_audio_frame_index_.store(0, std::memory_order_relaxed);
  const std::string key = f8::cppsdk::zenoh_data_key(cfg_.service_id, cfg_.service_id, "audio");
  auto publisher = std::make_shared<f8::cppsdk::ZenohLatestAudioChunkPublisher>();
  if (!publisher->open(runtime_backend, key)) {
    zenoh_audio_key_.clear();
    zenoh_audio_publisher_.reset();
    spdlog::error("audiocap zenoh audio publisher unavailable serviceId={} key={}", cfg_.service_id, key);
    bus_->stop();
    bus_.reset();
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    return false;
  }
  zenoh_audio_key_ = key;
  zenoh_audio_publisher_ = publisher;
  spdlog::info("audiocap zenoh audio publisher enabled serviceId={} key={}", cfg_.service_id, key);

  chunk_buffer_.assign(static_cast<std::size_t>(cfg_.frames_per_chunk) * cfg_.channels, 0.0f);
  capture_chunk_accum_.assign(static_cast<std::size_t>(cfg_.frames_per_chunk) * cfg_.channels, 0.0f);
  capture_accum_frames_ = 0;
  phase_ = 0.0;
  last_write_ms_ = 0;
  last_state_pub_ms_ = 0;
  last_device_refresh_ms_ = 0;
  selected_device_ = cfg_.device.empty() ? "Auto" : cfg_.device;
  std::string capture_error;
  if (!open_capture_device(selected_device_, capture_error)) {
    spdlog::error("audio capture device open failed: {}", capture_error);
    return false;
  }

  publish_static_state();
  publish_dynamic_state();

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("audiocap started serviceId={} backend={} audioBackend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend), "zenoh");
  return true;
}

void AudioCapService::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel) && !bus_ && !stream_ && !wasapi_) return;
  stop_requested_.store(true, std::memory_order_release);

  close_capture_device();

  if (zenoh_audio_publisher_) {
    zenoh_audio_publisher_->close();
  }
  zenoh_audio_publisher_.reset();
  zenoh_audio_key_.clear();
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
  if (now - last_device_refresh_ms_ >= 3000) {
    publish_state_if_changed("availableDevices", available_capture_devices());
    last_device_refresh_ms_ = now;
  }

  if (!active_.load(std::memory_order_acquire)) return;

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

  (void)write_audio_chunk_interleaved_f32(chunk_buffer_.data(), cfg_.frames_per_chunk, now);
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
      (void)write_audio_chunk_interleaved_f32(capture_chunk_accum_.data(), cfg_.frames_per_chunk, ts_ms);
      capture_accum_frames_ = 0;
    }
  }
}

bool AudioCapService::write_audio_chunk_interleaved_f32(const float* samples, std::uint32_t frames,
                                                        std::int64_t ts_ms) {
  if (samples == nullptr || frames == 0) {
    return false;
  }

  bool published_zenoh = false;
  auto publisher = zenoh_audio_publisher_;
  if (publisher && publisher->valid()) {
    const std::uint64_t seq = zenoh_audio_seq_.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::uint64_t frame_index =
        zenoh_audio_frame_index_.fetch_add(frames, std::memory_order_relaxed) + static_cast<std::uint64_t>(frames);
    const std::uint32_t bytes_per_frame = static_cast<std::uint32_t>(sizeof(float)) * static_cast<std::uint32_t>(cfg_.channels);
    f8::cppsdk::AudioChunkView chunk;
    chunk.sample_rate = cfg_.sample_rate;
    chunk.channels = static_cast<std::uint32_t>(cfg_.channels);
    chunk.format = f8::cppsdk::kAudioSampleFormatF32Le;
    chunk.frames = frames;
    chunk.bytes_per_frame = bytes_per_frame;
    chunk.seq = seq;
    chunk.frame_index = frame_index;
    chunk.ts_ms = ts_ms;
    chunk.payload = samples;
    chunk.payload_bytes = static_cast<std::size_t>(frames) * static_cast<std::size_t>(bytes_per_frame);
    published_zenoh = publisher->publish_chunk(chunk);
  }

  return published_zenoh;
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
  if (field != "selectedDevice") return;
  std::string ec;
  std::string em;
  if (!on_set_state(node_id, field, value, meta, ec, em)) {
    spdlog::warn("capture device state update rejected code={} reason={}", ec, em);
  }
}

bool AudioCapService::on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                                   const nlohmann::json& meta, std::string& error_code, std::string& error_message) {
  (void)meta;
  if (node_id != cfg_.service_id || field != "selectedDevice" || !value.is_string()) {
    error_code = "INVALID_ARGS";
    error_message = "expected selectedDevice string on the Audio Capture service node";
    return false;
  }
  const std::string selector = value.get<std::string>();
  if (selector == selected_device_) {
    error_code.clear();
    error_message.clear();
    return true;
  }
  const auto available = available_capture_devices();
  if (std::find(available.begin(), available.end(), selector) == available.end()) {
    error_code = "DEVICE_UNAVAILABLE";
    error_message = "capture device is not available: " + selector;
    return false;
  }

  const std::string previous = selected_device_;
  close_capture_device();
  if (!open_capture_device(selector, error_message)) {
    spdlog::error("capture device switch failed selector={} error={}", selector, error_message);
    std::string restore_error;
    if (!open_capture_device(previous, restore_error)) {
      spdlog::error("capture device restore failed selector={} error={}", previous, restore_error);
    }
    publish_static_state();
    error_code = "DEVICE_OPEN_FAILED";
    return false;
  }
  publish_static_state();
  error_code.clear();
  error_message.clear();
  spdlog::info("capture device switched selector={} activeDevice={}", selector, opened_device_name_);
  return true;
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
  publish_state_if_changed("serviceClass", cfg_.service_class);
  publish_state_if_changed("availableDevices", available_capture_devices());
  publish_state_if_changed("audioDevice", opened_device_name_);
  publish_state_if_changed("audioSampleRate", cfg_.sample_rate);
  publish_state_if_changed("audioChannels", cfg_.channels);
  publish_state_if_changed("audioFormat", "f32le");
  publish_state_if_changed("audioFramesPerChunk", cfg_.frames_per_chunk);
  publish_state_if_changed("audioChunkCount", cfg_.chunk_count);
  publish_state_if_changed("audioChunkSchemaVersion", 1);
  publish_state_if_changed("mode", cfg_.mode);
  publish_state_if_changed("toneHz", cfg_.tone_hz);
  publish_state_if_changed("gain", cfg_.gain);
}

void AudioCapService::publish_state_if_changed(const char* field, const nlohmann::json& value) {
  std::lock_guard<std::mutex> lock(state_mu_);
  const auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) (void)bus_->publish_state(cfg_.service_id, field, value, "audiocap", json::object());
}

void AudioCapService::publish_dynamic_state() {
  // Transport write sequence is intentionally not published as node state. It is
  // a high-frequency counter; consumers should read it from the active audio transport.
}

nlohmann::json AudioCapService::describe() {
  json spec;
  spec["service"] = {
      {"schemaVersion", "f8service/1"},
      {"serviceClass", "f8.audiocap"},
      {"label", "Audio Capture"},
      {"version", "0.0.1"},
      {"rendererClass", "default_svc"},
      {"tags", json::array({"audio", "capture", "zenoh"})},
      {"stateFields",
       json::array({
           state_field("availableDevices", json{{"type", "array"}, {"items", schema_string()}}, "ro",
                       "Available Devices", "Capture devices currently visible to the service.", false),
           state_field("selectedDevice", json{{"type", "string"}, {"default", "Auto"}}, "wo",
                       "Capture Device", "Device selected for audio capture.", true,
                       json{{"kind", "select"}, {"optionsFromState", "availableDevices"}}),
           state_field("audioDevice", schema_string(), "ro", "Audio Device", "Name of the audio capture device in use", false),
           state_field("audioSampleRate", schema_integer(), "ro", "Audio Sample Rate", "Sample rate of the audio capture device", false),
           state_field("audioChannels", schema_integer(), "ro", "Audio Channels", "Number of audio channels", false),
           state_field("audioFormat", schema_string(), "ro", "Audio Format", "Format of the audio data", false),
           state_field("audioFramesPerChunk", schema_integer(), "ro", "Audio Frames Per Chunk", "Number of audio frames per chunk", false),
           state_field("audioChunkCount", schema_integer(), "ro", "Audio Chunk Count", "Number of audio chunks", false),
           state_field("audioChunkSchemaVersion", schema_integer(), "ro", "Audio Chunk Schema Version",
                       "Zenoh audio chunk schema version.", false),
           state_field("mode", schema_string(), "rw", "Mode", "Current mode of the audio capture service", false),
           state_field("toneHz", schema_number(), "rw", "Tone Frequency", "Frequency of the generated tone", false),
           state_field("gain", schema_number(), "rw", "Gain", "Gain applied to the audio signal", false),
       })},
      {"commands", json::array()},
      {"dataInPorts", json::array()},
      {"dataOutPorts",
       json::array({
           audio_chunk_port("audio", "Captured audio chunk stream."),
       })},
  };
  spec["operators"] = json::array();
  return spec;
}

}  // namespace f8::audiocap
