#include "wasapi_loopback_capture.h"

#if defined(_WIN32)

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <vector>

#include <Windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <propsys.h>
#include <functiondiscoverykeys_devpkey.h>
#include <propidl.h>
#include <wrl/client.h>

#include <spdlog/spdlog.h>

#include "f8cppsdk/time_utils.h"

namespace f8::audiocap {

using Microsoft::WRL::ComPtr;

namespace {

std::string hr_to_string(HRESULT hr) {
  char* msg = nullptr;
  const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
  const DWORD n = FormatMessageA(flags, nullptr, static_cast<DWORD>(hr), 0, reinterpret_cast<LPSTR>(&msg), 0, nullptr);
  std::string out;
  if (n && msg) {
    out.assign(msg, msg + n);
    while (!out.empty() && (out.back() == '\r' || out.back() == '\n')) out.pop_back();
  } else {
    out = "HRESULT=0x" + std::to_string(static_cast<std::uint32_t>(hr));
  }
  if (msg) LocalFree(msg);
  return out;
}

int sdl_bytes_per_sample(SDL_AudioFormat fmt) {
  switch (fmt) {
    case SDL_AUDIO_F32:
    case SDL_AUDIO_S32:
      return 4;
    case SDL_AUDIO_S16:
      return 2;
    default:
      return 0;
  }
}

bool waveformat_to_sdl_spec(const WAVEFORMATEX* wfx, SDL_AudioSpec& out) {
  if (!wfx) return false;
  out.freq = static_cast<int>(wfx->nSamplesPerSec);
  out.channels = static_cast<int>(wfx->nChannels);

  auto set_fmt = [&](SDL_AudioFormat fmt) {
    out.format = fmt;
    return true;
  };

  if (wfx->wFormatTag == WAVE_FORMAT_IEEE_FLOAT && wfx->wBitsPerSample == 32) {
    return set_fmt(SDL_AUDIO_F32);
  }

  if (wfx->wFormatTag == WAVE_FORMAT_PCM) {
    if (wfx->wBitsPerSample == 16) return set_fmt(SDL_AUDIO_S16);
    if (wfx->wBitsPerSample == 32) return set_fmt(SDL_AUDIO_S32);
    return false;
  }

  if (wfx->wFormatTag == WAVE_FORMAT_EXTENSIBLE && wfx->cbSize >= sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX)) {
    const auto* ext = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wfx);
    if (ext->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT && wfx->wBitsPerSample == 32) {
      return set_fmt(SDL_AUDIO_F32);
    }
    if (ext->SubFormat == KSDATAFORMAT_SUBTYPE_PCM) {
      if (wfx->wBitsPerSample == 16) return set_fmt(SDL_AUDIO_S16);
      if (wfx->wBitsPerSample == 32) return set_fmt(SDL_AUDIO_S32);
      return false;
    }
  }

  return false;
}

std::string get_device_friendly_name(IMMDevice* device) {
  if (!device) return {};
  ComPtr<IPropertyStore> props;
  if (FAILED(device->OpenPropertyStore(STGM_READ, &props)) || !props) return {};

  PROPVARIANT v;
  PropVariantInit(&v);
  std::string out;
  if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &v)) && v.vt == VT_LPWSTR && v.pwszVal) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, v.pwszVal, -1, nullptr, 0, nullptr, nullptr);
    if (n > 1) {
      out.resize(static_cast<std::size_t>(n - 1));
      (void)WideCharToMultiByte(CP_UTF8, 0, v.pwszVal, -1, out.data(), n, nullptr, nullptr);
    }
  }
  PropVariantClear(&v);
  return out;
}

}  // namespace

WasapiLoopbackCapture::WasapiLoopbackCapture(Config cfg) : cfg_(cfg) {}

WasapiLoopbackCapture::~WasapiLoopbackCapture() { stop(); }

bool WasapiLoopbackCapture::start(Callback cb, std::string& out_device_name, std::string& out_error) {
  if (running_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(init_mu_);
    out_device_name = device_name_;
    out_error.clear();
    return true;
  }

  cb_ = std::move(cb);
  {
    std::lock_guard<std::mutex> lock(init_mu_);
    device_name_.clear();
    error_.clear();
    init_done_ = false;
  }

  stop_requested_.store(false, std::memory_order_release);
  paused_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_release);

  worker_ = std::thread([this] { thread_main(); });

  {
    std::unique_lock<std::mutex> lock(init_mu_);
    const bool ok = init_cv_.wait_for(lock, std::chrono::milliseconds(3000), [this] { return init_done_; });
    out_device_name = device_name_;
    out_error = error_;
    if (!ok) out_error = "WASAPI init timeout";
  }

  if (!out_error.empty()) {
    stop();
    return false;
  }
  return true;
}

void WasapiLoopbackCapture::stop() {
  stop_requested_.store(true, std::memory_order_release);
  running_.store(false, std::memory_order_release);
  if (worker_.joinable()) worker_.join();
  cb_ = nullptr;
}

void WasapiLoopbackCapture::thread_main() {
  auto on_exit = [this] { running_.store(false, std::memory_order_release); };

  auto set_init_error = [&](const std::string& msg) {
    {
      std::lock_guard<std::mutex> lock(init_mu_);
      error_ = msg;
      init_done_ = true;
    }
    init_cv_.notify_all();
  };

  auto set_init_ok = [&](const std::string& dev) {
    {
      std::lock_guard<std::mutex> lock(init_mu_);
      device_name_ = dev;
      init_done_ = true;
    }
    init_cv_.notify_all();
  };

  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  const bool co_inited = SUCCEEDED(hr);
  if (!co_inited) {
    const std::string msg = "CoInitializeEx failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    on_exit();
    return;
  }

  ComPtr<IMMDeviceEnumerator> enumerator;
  hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                        reinterpret_cast<void**>(enumerator.GetAddressOf()));
  if (FAILED(hr) || !enumerator) {
    const std::string msg = "CoCreateInstance(MMDeviceEnumerator) failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CoUninitialize();
    on_exit();
    return;
  }

  ComPtr<IMMDevice> device;
  hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
  if (FAILED(hr) || !device) {
    const std::string msg = "GetDefaultAudioEndpoint(eRender) failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CoUninitialize();
    on_exit();
    return;
  }

  std::string device_name = get_device_friendly_name(device.Get());
  if (device_name.empty()) device_name = "WASAPI default render (loopback)";

  ComPtr<IAudioClient> audio_client;
  hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(audio_client.GetAddressOf()));
  if (FAILED(hr) || !audio_client) {
    const std::string msg = "IMMDevice::Activate(IAudioClient) failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CoUninitialize();
    on_exit();
    return;
  }

  WAVEFORMATEX* mix = nullptr;
  hr = audio_client->GetMixFormat(&mix);
  if (FAILED(hr) || !mix) {
    const std::string msg = "IAudioClient::GetMixFormat failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CoUninitialize();
    on_exit();
    return;
  }

  const int src_block_align = static_cast<int>(mix->nBlockAlign);

  SDL_AudioSpec src_spec{};
  SDL_AudioSpec dst_spec{};
  dst_spec.freq = static_cast<int>(cfg_.dst_sample_rate);
  dst_spec.channels = static_cast<int>(cfg_.dst_channels);
  dst_spec.format = SDL_AUDIO_F32;

  if (!waveformat_to_sdl_spec(mix, src_spec)) {
    const std::string msg = "unsupported WASAPI mix format";
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CoTaskMemFree(mix);
    CoUninitialize();
    on_exit();
    return;
  }

  SDL_AudioStream* conv = SDL_CreateAudioStream(&src_spec, &dst_spec);
  if (!conv) {
    const std::string msg = std::string("SDL_CreateAudioStream failed: ") + SDL_GetError();
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CoTaskMemFree(mix);
    CoUninitialize();
    on_exit();
    return;
  }

  const REFERENCE_TIME hns_buffer_duration = 10000000 / 10;  // 100ms
  const DWORD flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
  hr = audio_client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, hns_buffer_duration, 0, mix, nullptr);
  CoTaskMemFree(mix);
  mix = nullptr;
  if (FAILED(hr)) {
    const std::string msg = "IAudioClient::Initialize(loopback) failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    SDL_DestroyAudioStream(conv);
    CoUninitialize();
    on_exit();
    return;
  }

  HANDLE capture_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!capture_event) {
    const std::string msg = "CreateEventW failed";
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    SDL_DestroyAudioStream(conv);
    CoUninitialize();
    on_exit();
    return;
  }

  hr = audio_client->SetEventHandle(capture_event);
  if (FAILED(hr)) {
    const std::string msg = "IAudioClient::SetEventHandle failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CloseHandle(capture_event);
    SDL_DestroyAudioStream(conv);
    CoUninitialize();
    on_exit();
    return;
  }

  ComPtr<IAudioCaptureClient> capture_client;
  hr = audio_client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(capture_client.GetAddressOf()));
  if (FAILED(hr) || !capture_client) {
    const std::string msg = "IAudioClient::GetService(IAudioCaptureClient) failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CloseHandle(capture_event);
    SDL_DestroyAudioStream(conv);
    CoUninitialize();
    on_exit();
    return;
  }

  hr = audio_client->Start();
  if (FAILED(hr)) {
    const std::string msg = "IAudioClient::Start failed: " + hr_to_string(hr);
    spdlog::error("WASAPI loopback: {}", msg);
    set_init_error(msg);
    CloseHandle(capture_event);
    SDL_DestroyAudioStream(conv);
    CoUninitialize();
    on_exit();
    return;
  }

  set_init_ok(device_name);

  spdlog::info("WASAPI loopback started device=\"{}\" src={}Hz/{}ch dst={}Hz/{}ch", device_name, src_spec.freq,
               src_spec.channels, dst_spec.freq, dst_spec.channels);

  const int out_frame_bytes = static_cast<int>(sizeof(float) * cfg_.dst_channels);
  std::vector<float> out_tmp;
  std::vector<std::uint8_t> silent_bytes;
  const int src_frame_bytes = src_block_align > 0 ? src_block_align
                                                  : (sdl_bytes_per_sample(src_spec.format) * src_spec.channels);

  while (!stop_requested_.load(std::memory_order_acquire)) {
    if (paused_.load(std::memory_order_acquire)) {
      // Keep draining WASAPI buffers to avoid backpressure.
      UINT32 packet = 0;
      if (SUCCEEDED(capture_client->GetNextPacketSize(&packet)) && packet > 0) {
        BYTE* data = nullptr;
        UINT32 frames = 0;
        DWORD buf_flags = 0;
        if (SUCCEEDED(capture_client->GetBuffer(&data, &frames, &buf_flags, nullptr, nullptr))) {
          capture_client->ReleaseBuffer(frames);
        }
      }
      Sleep(1);
      continue;
    }

    const DWORD wait = WaitForSingleObject(capture_event, 200);
    if (wait != WAIT_OBJECT_0) continue;  // timeout or unexpected

    UINT32 packet_length = 0;
    hr = capture_client->GetNextPacketSize(&packet_length);
    if (FAILED(hr)) continue;

    while (packet_length != 0) {
      BYTE* data = nullptr;
      UINT32 frames = 0;
      DWORD buf_flags = 0;
      hr = capture_client->GetBuffer(&data, &frames, &buf_flags, nullptr, nullptr);
      if (FAILED(hr)) break;

      const std::int64_t ts_ms = f8::cppsdk::now_ms();
      const int in_bytes = static_cast<int>(frames) * src_frame_bytes;

      const void* put_ptr = data;
      if (buf_flags & AUDCLNT_BUFFERFLAGS_SILENT) {
        if (silent_bytes.size() < static_cast<std::size_t>(in_bytes)) silent_bytes.resize(static_cast<std::size_t>(in_bytes));
        std::memset(silent_bytes.data(), 0, static_cast<std::size_t>(in_bytes));
        put_ptr = silent_bytes.data();
      }

      if (in_bytes > 0) {
        if (!SDL_PutAudioStreamData(conv, put_ptr, in_bytes)) {
          spdlog::warn("WASAPI loopback: SDL_PutAudioStreamData failed: {}", SDL_GetError());
        }
      }

      capture_client->ReleaseBuffer(frames);

      for (;;) {
        const int avail = SDL_GetAudioStreamAvailable(conv);
        if (avail < out_frame_bytes) break;
        const int want = std::min(avail, 8192 * out_frame_bytes);
        const int aligned = (want / out_frame_bytes) * out_frame_bytes;
        if (aligned <= 0) break;
        out_tmp.resize(static_cast<std::size_t>(aligned / static_cast<int>(sizeof(float))));
        const int got = SDL_GetAudioStreamData(conv, out_tmp.data(), aligned);
        if (got <= 0) break;
        const int got_aligned = (got / out_frame_bytes) * out_frame_bytes;
        const std::uint32_t out_frames = static_cast<std::uint32_t>(got_aligned / out_frame_bytes);
        if (out_frames > 0 && cb_) cb_(out_tmp.data(), out_frames, ts_ms);
      }

      hr = capture_client->GetNextPacketSize(&packet_length);
      if (FAILED(hr)) break;
    }
  }

  (void)audio_client->Stop();
  CloseHandle(capture_event);
  SDL_DestroyAudioStream(conv);
  CoUninitialize();
  on_exit();
}

}  // namespace f8::audiocap

#else

namespace f8::audiocap {

WasapiLoopbackCapture::WasapiLoopbackCapture(Config) {}
WasapiLoopbackCapture::~WasapiLoopbackCapture() = default;
bool WasapiLoopbackCapture::start(Callback, std::string& out_device_name, std::string& out_error) {
  out_device_name.clear();
  out_error = "WASAPI loopback is only available on Windows";
  return false;
}
void WasapiLoopbackCapture::stop() {}

}  // namespace f8::audiocap

#endif
