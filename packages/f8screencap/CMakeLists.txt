add_executable(f8screencap_service
  src/main.cpp
  src/screencap_service.cpp
  src/win32_capture_sources.cpp
  src/win32_picker.cpp
  src/win32_wgc_capture.cpp
)

target_compile_features(f8screencap_service PRIVATE cxx_std_17)

target_include_directories(f8screencap_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

target_link_libraries(f8screencap_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

target_compile_definitions(f8screencap_service PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

if (WIN32)
  target_link_libraries(f8screencap_service PRIVATE user32 gdi32 dwmapi d3d11 dxgi windowsapp)
endif()
