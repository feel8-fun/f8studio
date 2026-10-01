add_executable(f8cppsdk_tests
  test_main_thread_queue.cpp
  test_protocol_models_parse.cpp
  test_cpp_runtime_host_exec.cpp
  test_rungraph_routes.cpp
  test_service_bus_integration.cpp
)

target_compile_features(f8cppsdk_tests PRIVATE cxx_std_17)
target_compile_definitions(f8cppsdk_tests PRIVATE F8_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

find_package(GTest REQUIRED)

target_link_libraries(f8cppsdk_tests
  PRIVATE
    f8cppsdk
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME f8cppsdk_tests COMMAND f8cppsdk_tests)
