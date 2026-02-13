if(NOT F8_ENABLE_DLINFER)
  return()
endif()

find_package(cxxopts REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

add_executable(f8dlinfer_detector_service
  src/main.cpp
)

target_compile_features(f8dlinfer_detector_service PRIVATE cxx_std_17)
target_compile_definitions(f8dlinfer_detector_service PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(f8dlinfer_detector_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

f8_deploy_service_runtime(f8dlinfer_detector_service "f8/dl/detector_cpp")
