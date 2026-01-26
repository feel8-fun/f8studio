add_executable(f8audiocap_service
  src/main.cpp
  src/audiocap_service.cpp
  src/wasapi_loopback_capture.cpp
)

target_compile_features(f8audiocap_service PRIVATE cxx_std_17)

target_include_directories(f8audiocap_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(SDL3 REQUIRED)

target_link_libraries(f8audiocap_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    sdl::sdl
)

target_compile_definitions(f8audiocap_service PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

if (WIN32)
  target_link_libraries(f8audiocap_service PRIVATE ole32 uuid avrt)
endif()
