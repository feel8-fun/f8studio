add_library(f8cvkit
  src/base64.cpp
  src/cvkit_image_io.cpp
)

target_compile_features(f8cvkit PUBLIC cxx_std_17)

target_include_directories(f8cvkit
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenCV REQUIRED CONFIG)
find_package(spdlog REQUIRED)
find_package(cxxopts REQUIRED)
find_package(nlohmann_json REQUIRED)

target_link_libraries(f8cvkit
  PUBLIC
    ${OpenCV_LIBS}
  PRIVATE
    spdlog::spdlog
)

add_executable(f8cvkit_template_match_service
  src/services/template_match/main.cpp
  src/services/template_match/template_match_service.cpp
)

add_executable(f8cvkit_dense_optflow_service
  src/services/dense_optflow/main.cpp
  src/services/dense_optflow/dense_optflow_service.cpp
)

add_executable(f8cvkit_tracking_service
  src/services/tracking/main.cpp
  src/services/tracking/tracking_service.cpp
)

add_executable(f8cvkit_video_stab_service
  src/services/video_stab/main.cpp
  src/services/video_stab/video_stab_service.cpp
)

foreach(_t IN ITEMS f8cvkit_template_match_service f8cvkit_dense_optflow_service f8cvkit_tracking_service f8cvkit_video_stab_service)
  target_compile_features(${_t} PRIVATE cxx_std_17)
  target_link_libraries(${_t}
    PRIVATE
      f8cppsdk
      f8cvkit
      cxxopts::cxxopts
      spdlog::spdlog
      nlohmann_json::nlohmann_json
  )
  target_compile_definitions(${_t} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Deploy all CVKit services into a shared directory so they can share runtime deps.
  f8_deploy_service_runtime(${_t} "f8/cvkit" NO_CLEAN)
endforeach()
