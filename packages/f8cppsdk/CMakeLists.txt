add_library(f8cppsdk
  src/nats_client.cpp
  src/kv_store.cpp
  src/service_control_plane.cpp
  src/service_bus.cpp
  src/state_kv.cpp
  src/data_bus.cpp
  src/rungraph_routes.cpp
  src/time_utils.cpp
  src/f8_naming.cpp
  src/shm_region.cpp
  src/shm_event_wait.cpp
  src/video_shared_memory_sink.cpp
  src/audio_shared_memory_sink.cpp
)

target_include_directories(f8cppsdk
  PUBLIC
    # Prefer generated headers from the build tree over any source-tree remnants.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(f8cppsdk PUBLIC cxx_std_17)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Protocol model codegen (OpenAPI -> C++ header), similar to pysdk datamodel-code-generator.
set(F8CPP_PROTOCOL_MODELS_OUT
    ${CMAKE_CURRENT_BINARY_DIR}/generated/f8cppsdk/generated/protocol_models.h
)
add_custom_command(
  OUTPUT ${F8CPP_PROTOCOL_MODELS_OUT}
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/scripts/gen_cpp_protocol_models.py
          --protocol ${CMAKE_SOURCE_DIR}/schemas/protocol.yml
          --out ${F8CPP_PROTOCOL_MODELS_OUT}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/scripts/gen_cpp_protocol_models.py
    ${CMAKE_SOURCE_DIR}/schemas/protocol.yml
  VERBATIM
)
add_custom_target(f8cppsdk_codegen_protocol_models DEPENDS ${F8CPP_PROTOCOL_MODELS_OUT})
add_dependencies(f8cppsdk f8cppsdk_codegen_protocol_models)

find_package(cnats REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

target_link_libraries(f8cppsdk
  PUBLIC
    cnats::nats_static
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
