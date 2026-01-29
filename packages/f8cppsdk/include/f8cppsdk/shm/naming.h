#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace f8::cppsdk::shm {

inline std::string video_shm_name(const std::string& service_id) { return "shm." + service_id + ".video"; }
inline std::string audio_shm_name(const std::string& service_id) { return "shm." + service_id + ".audio"; }

inline std::string frame_event_name(const std::string& shm_name) { return shm_name + "_evt"; }

inline constexpr std::size_t kDefaultVideoShmBytes = 256ull * 1024ull * 1024ull;
inline constexpr std::uint32_t kDefaultVideoShmSlots = 2;

inline constexpr std::size_t kDefaultAudioShmBytes = 8ull * 1024ull * 1024ull;

}  // namespace f8::cppsdk::shm

