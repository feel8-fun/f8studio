add_executable(f8sdk_demo_service
  src/main.cpp
  src/demo_service.cpp
)

target_compile_features(f8sdk_demo_service PRIVATE cxx_std_17)

target_include_directories(f8sdk_demo_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

target_link_libraries(f8sdk_demo_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

target_compile_definitions(f8sdk_demo_service PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

