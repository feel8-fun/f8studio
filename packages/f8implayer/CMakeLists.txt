add_executable(f8implayer_service
  src/main.cpp
  src/implayer_service.cpp
  src/implayer_gui.cpp
  src/sdl_video_window.cpp
  src/gl_loader.cpp
  src/mpv_player.cpp
  src/imgui_backends/imgui_impl_sdl3.cpp
  src/imgui_backends/imgui_impl_opengl3.cpp
)

target_compile_features(f8implayer_service PRIVATE cxx_std_17)

target_include_directories(f8implayer_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glad REQUIRED)
find_package(libmpv REQUIRED)
find_package(opengl_system REQUIRED)
find_package(imgui REQUIRED)
find_package(iconfontcppheaders REQUIRED CONFIG)

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
    iconfontcppheaders::iconfontcppheaders
)

target_compile_definitions(f8implayer_service PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD NOMINMAX WIN32_LEAN_AND_MEAN)

# Copy font assets next to the executable for local runs, and into the deployed
# services directory for packaged/runtime use.
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
  VERBATIM
  COMMENT "Copy f8implayer font assets to build output"
)

add_custom_command(
  TARGET f8implayer_service
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${F8_SERVICES_ROOT}/f8/implayer/${_implayer_platform_dir}/font"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/font"
          "${F8_SERVICES_ROOT}/f8/implayer/${_implayer_platform_dir}/font"
  VERBATIM
  COMMENT "Copy f8implayer font assets to services deploy dir"
)

# Deploy C++ service artifacts into `services/.../<platform>/` for packaging/runtime use.
f8_deploy_service_runtime(f8implayer_service "f8/implayer")
