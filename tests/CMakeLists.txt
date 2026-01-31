add_executable(f8cppsdk_tests
  test_main_thread_queue.cpp
  test_protocol_models_parse.cpp
  test_rungraph_routes.cpp
)

target_compile_features(f8cppsdk_tests PRIVATE cxx_std_17)

find_package(GTest REQUIRED)

target_link_libraries(f8cppsdk_tests
  PRIVATE
    f8cppsdk
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME f8cppsdk_tests COMMAND f8cppsdk_tests)
