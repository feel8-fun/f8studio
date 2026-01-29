add_library(f8cppsdk
  src/nats_client.cpp
  src/kv_store.cpp
  src/service_control_plane.cpp
  src/service_bus.cpp
  src/state_kv.cpp
  src/data_bus.cpp
  src/time_utils.cpp
  src/f8_naming.cpp
  src/shm_region.cpp
  src/shm_event_wait.cpp
  src/video_shared_memory_sink.cpp
  src/audio_shared_memory_sink.cpp
)

target_include_directories(f8cppsdk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(f8cppsdk PUBLIC cxx_std_17)

find_package(cnats REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

target_link_libraries(f8cppsdk
  PUBLIC
    cnats::nats_static
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
