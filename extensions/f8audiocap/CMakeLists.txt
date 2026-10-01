cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(f8audiocap VERSION 0.0.1 LANGUAGES CXX)
    find_package(f8cppsdk 0.1 CONFIG REQUIRED)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

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

if (TARGET SDL3::SDL3)
  set(F8_SDL3_TARGET SDL3::SDL3)
elseif (TARGET sdl::sdl)
  set(F8_SDL3_TARGET sdl::sdl)
else()
  message(FATAL_ERROR "SDL3 package did not provide SDL3::SDL3 or sdl::sdl")
endif()

target_link_libraries(f8audiocap_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${F8_SDL3_TARGET}
)

target_compile_definitions(f8audiocap_service PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

if (WIN32)
  target_link_libraries(f8audiocap_service PRIVATE ole32 uuid avrt)
endif()

# Deploy C++ service artifacts into `runtime/bundles/<bundle>/<version>/<platform>/` for packaging/runtime use.
f8_deploy_service_runtime(f8audiocap_service "f8.audiocap")
