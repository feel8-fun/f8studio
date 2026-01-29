add_executable(f8screencap_service
  src/main.cpp
  src/screencap_service.cpp
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

if (WIN32)
  target_sources(f8screencap_service PRIVATE
    src/win32_capture_sources.cpp
    src/win32_picker.cpp
    src/win32_wgc_capture.cpp
  )
  target_compile_definitions(f8screencap_service PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(f8screencap_service PRIVATE user32 gdi32 dwmapi d3d11 dxgi windowsapp)
else()
  target_sources(f8screencap_service PRIVATE
    src/linux_x11_capture.cpp
    src/x11_capture_sources.cpp
    src/x11_picker.cpp
  )
  if (UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(f8screencap_service PRIVATE X11::X11)
  endif()
endif()
