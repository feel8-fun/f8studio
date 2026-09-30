include(ExternalProject)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(F8_IMPLAYER_LIBMPV_WIN_URL
    "https://downloads.sourceforge.net/mpv-player-windows/mpv-dev-x86_64-20250713-git-bd21180.7z")
set(F8_IMPLAYER_LIBMPV_WIN_SHA256 "e75982d90a1fa3620b194204420386f590b4e96f03dc9daf46b996fccce3549f")

set(F8_IMPLAYER_SOURCES
  src/main.cpp
  src/implayer_service.cpp
  src/implayer_gui.cpp
  src/sdl_video_window.cpp
  src/gl_loader.cpp
  src/mpv_player.cpp
  src/openxr_presenter.cpp
  src/imgui_backends/imgui_impl_sdl3.cpp
  src/imgui_backends/imgui_impl_opengl3.cpp
)

if(WIN32)
  list(APPEND F8_IMPLAYER_SOURCES src/implayer.rc)
endif()

add_executable(f8implayer_service ${F8_IMPLAYER_SOURCES})

target_compile_features(f8implayer_service PRIVATE cxx_std_17)

target_include_directories(f8implayer_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openxr/include
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glad REQUIRED)
find_package(opengl_system REQUIRED)
find_package(imgui REQUIRED)
find_package(iconfontcppheaders REQUIRED CONFIG)
if(NOT WIN32)
  find_package(PNG REQUIRED)
endif()

if (TARGET SDL3::SDL3)
  set(F8_SDL3_TARGET SDL3::SDL3)
elseif (TARGET sdl::sdl)
  set(F8_SDL3_TARGET sdl::sdl)
else()
  message(FATAL_ERROR "SDL3 package did not provide SDL3::SDL3 or sdl::sdl")
endif()

set(_implayer_ytdlp_hints "")
if(DEFINED F8_PIXI_CPP_ENV_DIR)
  list(APPEND _implayer_ytdlp_hints
    "${F8_PIXI_CPP_ENV_DIR}/bin"
    "${F8_PIXI_CPP_ENV_DIR}/Scripts"
  )
endif()
find_program(
  F8_IMPLAYER_YTDLP_EXE
  NAMES yt-dlp.exe yt-dlp
  HINTS ${_implayer_ytdlp_hints}
  REQUIRED
  DOC "yt-dlp executable from the pixi/conda cpp environment"
)

if(WIN32)
  set(_implayer_mpv_root "${CMAKE_CURRENT_BINARY_DIR}/third_party/mpv-dev")
  set(_implayer_mpv_include "${_implayer_mpv_root}/include")
  set(_implayer_mpv_dll "${_implayer_mpv_root}/libmpv-2.dll")
  set(_implayer_mpv_implib "${_implayer_mpv_root}/libmpv.dll.a")

  ExternalProject_Add(f8_implayer_mpv_dev
    URL "${F8_IMPLAYER_LIBMPV_WIN_URL}"
    URL_HASH "SHA256=${F8_IMPLAYER_LIBMPV_WIN_SHA256}"
    DOWNLOAD_NO_PROGRESS ON
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      "${CMAKE_COMMAND}" -E make_directory "${_implayer_mpv_root}/include"
      COMMAND "${CMAKE_COMMAND}" -E copy_directory <SOURCE_DIR>/include "${_implayer_mpv_root}/include"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different <SOURCE_DIR>/libmpv-2.dll "${_implayer_mpv_dll}"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different <SOURCE_DIR>/libmpv.dll.a "${_implayer_mpv_implib}"
  )

  add_library(f8_implayer_mpv INTERFACE)
  add_dependencies(f8_implayer_mpv f8_implayer_mpv_dev)
  target_include_directories(f8_implayer_mpv INTERFACE "${_implayer_mpv_include}")
  target_link_libraries(f8_implayer_mpv INTERFACE "${_implayer_mpv_implib}")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(MPV REQUIRED IMPORTED_TARGET mpv)

  add_library(f8_implayer_mpv INTERFACE)
  target_link_libraries(f8_implayer_mpv INTERFACE PkgConfig::MPV)
endif()

target_link_libraries(f8implayer_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${F8_SDL3_TARGET}
    glad::glad
    f8_implayer_mpv
    opengl::opengl
    imgui::imgui
    iconfontcppheaders::iconfontcppheaders
)

if(NOT WIN32)
  target_link_libraries(f8implayer_service PRIVATE PNG::PNG)
endif()

target_compile_definitions(f8implayer_service PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD NOMINMAX WIN32_LEAN_AND_MEAN)

# Copy runtime assets next to the executable for local runs, and into the
# deployed services directory for packaged/runtime use.
if(WIN32)
  set(_implayer_platform_dir "win")
elseif(APPLE)
  set(_implayer_platform_dir "mac")
else()
  set(_implayer_platform_dir "linux")
endif()

add_custom_command(
  TARGET f8implayer_service
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:f8implayer_service>/font"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/font"
          "$<TARGET_FILE_DIR:f8implayer_service>/font"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:f8implayer_service>/resources"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/resources"
          "$<TARGET_FILE_DIR:f8implayer_service>/resources"
  VERBATIM
  COMMENT "Copy f8implayer runtime assets to build output"
)

if(WIN32)
  add_dependencies(f8implayer_service f8_implayer_mpv_dev)
  add_custom_command(
    TARGET f8implayer_service
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_implayer_mpv_dll}" "$<TARGET_FILE_DIR:f8implayer_service>/libmpv-2.dll"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "$<TARGET_FILE_DIR:f8implayer_service>/yt-dlp.exe"
    VERBATIM
    COMMENT "Copy libmpv runtime binary to build output"
  )
endif()

add_custom_command(
  TARGET f8implayer_service
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${F8_SERVICE_BUNDLES_ROOT}/f8.implayer/0.0.1/${_implayer_platform_dir}/font"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/font"
          "${F8_SERVICE_BUNDLES_ROOT}/f8.implayer/0.0.1/${_implayer_platform_dir}/font"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${F8_SERVICE_BUNDLES_ROOT}/f8.implayer/0.0.1/${_implayer_platform_dir}/resources"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/resources"
          "${F8_SERVICE_BUNDLES_ROOT}/f8.implayer/0.0.1/${_implayer_platform_dir}/resources"
  VERBATIM
  COMMENT "Copy f8implayer runtime assets to bundle deploy dir"
)

if(WIN32)
  add_custom_command(
    TARGET f8implayer_service
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_implayer_mpv_dll}" "${F8_SERVICE_BUNDLES_ROOT}/f8.implayer/0.0.1/${_implayer_platform_dir}/libmpv-2.dll"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${F8_SERVICE_BUNDLES_ROOT}/f8.implayer/0.0.1/${_implayer_platform_dir}/yt-dlp.exe"
    VERBATIM
    COMMENT "Copy libmpv runtime binary to bundle deploy dir"
  )
endif()

# Deploy C++ service artifacts into `runtime/bundles/<bundle>/<version>/<platform>/` for packaging/runtime use.
f8_deploy_service_runtime(f8implayer_service "f8.implayer")
