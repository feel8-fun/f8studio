add_executable(f8implayer_service
  src/main.cpp
  src/implayer_service.cpp
  src/implayer_gui.cpp
  src/video_shared_memory_sink.cpp
  src/shm_region.cpp
  src/sdl_video_window.cpp
  src/gl_loader.cpp
  src/mpv_player.cpp
  src/imgui_backends/imgui_impl_sdl3.cpp
  src/imgui_backends/imgui_impl_opengl3.cpp
  src/ffmpeg_decoder.cpp
)

target_compile_features(f8implayer_service PRIVATE cxx_std_17)

target_include_directories(f8implayer_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ffmpeg REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glad REQUIRED)
find_package(libmpv REQUIRED)
find_package(opengl_system REQUIRED)
find_package(imgui REQUIRED)

# Conan's ffmpeg package currently does not populate ffmpeg_INCLUDE_DIRS for MSVC;
# add the package include directory explicitly.
if(DEFINED ffmpeg_PACKAGE_FOLDER_RELEASE)
  target_include_directories(f8implayer_service PRIVATE "$<$<CONFIG:Release>:${ffmpeg_PACKAGE_FOLDER_RELEASE}/include>")
endif()
if(DEFINED ffmpeg_PACKAGE_FOLDER_DEBUG)
  target_include_directories(f8implayer_service PRIVATE "$<$<CONFIG:Debug>:${ffmpeg_PACKAGE_FOLDER_DEBUG}/include>")
endif()

target_link_libraries(f8implayer_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    sdl::sdl
    glad::glad
    libmpv::libmpv
    opengl::opengl
    imgui::imgui
    ffmpeg::avformat
    ffmpeg::avcodec
    ffmpeg::avutil
    ffmpeg::swscale
)

target_compile_definitions(f8implayer_service PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD NOMINMAX WIN32_LEAN_AND_MEAN)
