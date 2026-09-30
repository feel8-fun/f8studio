add_library(f8cppsdk
  src/describe_schema.cpp
  src/exec_flow_executor.cpp
  src/service_bus.cpp
  src/service_host.cpp
  src/runtime_node.cpp
  src/runtime_node_registry.cpp
  src/runtime_backend.cpp
  src/zenoh_naming.cpp
  src/zenoh_transport.cpp
  src/msg_codec.cpp
  src/rungraph_routes.cpp
  src/time_utils.cpp
  src/f8_naming.cpp
  src/describe_builtins.cpp
  src/latest_binary_stream_transport.cpp
  src/latest_audio_chunk_transport.cpp
  src/latest_video_frame_transport.cpp
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

# Checked-in stream codecs are shared with Python; fail on schema drift.
add_custom_target(f8cppsdk_check_stream_wire
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_stream_wire.py --check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_runtime_keys.py --check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_runtime_policy.py --check
  VERBATIM
)
add_dependencies(f8cppsdk f8cppsdk_check_stream_wire)

find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

set(_f8_zenoh_link_targets "")
if(F8_WITH_ZENOH)
  find_package(zenohc CONFIG REQUIRED)
  set(_f8_zenoh_have_cxx FALSE)
  set(_f8_zenoh_have_c FALSE)
  if(TARGET zenohc::zenohc)
    set(_f8_zenoh_have_c TRUE)
    list(APPEND _f8_zenoh_link_targets zenohc::zenohc)
  elseif(TARGET zenohc::lib)
    set(_f8_zenoh_have_c TRUE)
    list(APPEND _f8_zenoh_link_targets zenohc::lib)
  endif()
  if(_f8_zenoh_have_c)
    find_package(zenohcxx CONFIG REQUIRED)
  else()
    message(FATAL_ERROR "F8_WITH_ZENOH=ON requires a zenohc CMake target (expected zenohc::zenohc or zenohc::lib).")
  endif()
  if(TARGET zenohcxx::zenohc)
    set(_f8_zenoh_have_cxx TRUE)
    list(APPEND _f8_zenoh_link_targets zenohcxx::zenohc)
  elseif(TARGET zenohcxx)
    set(_f8_zenoh_have_cxx TRUE)
    list(APPEND _f8_zenoh_link_targets zenohcxx)
  endif()
  if(_f8_zenoh_have_cxx AND _f8_zenoh_have_c)
    target_compile_definitions(f8cppsdk PUBLIC F8_WITH_ZENOH=1)
    target_link_libraries(f8cppsdk PUBLIC ${_f8_zenoh_link_targets})
  else()
    message(FATAL_ERROR "F8_WITH_ZENOH=ON requires a zenohcxx CMake target (expected zenohcxx::zenohc or zenohcxx).")
  endif()
else()
  target_compile_definitions(f8cppsdk PUBLIC F8_WITH_ZENOH=0)
endif()

target_link_libraries(f8cppsdk
  PUBLIC
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

add_executable(f8cpp_bench_latest_video_transport EXCLUDE_FROM_ALL
  bench/bench_latest_video_transport.cpp
)
target_link_libraries(f8cpp_bench_latest_video_transport PRIVATE f8cppsdk)

add_executable(f8cpp_crosslang_video_publisher EXCLUDE_FROM_ALL
  bench/crosslang_video_publisher.cpp
)
target_link_libraries(f8cpp_crosslang_video_publisher PRIVATE f8cppsdk)

add_executable(f8cpp_crosslang_runtime_probe EXCLUDE_FROM_ALL
  bench/crosslang_runtime_probe.cpp
)
target_link_libraries(f8cpp_crosslang_runtime_probe PRIVATE f8cppsdk)
