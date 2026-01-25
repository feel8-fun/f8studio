add_executable(f8webrtc_gateway_service
  src/main.cpp
  src/webrtc_gateway_service.cpp
  src/ws_signaling_server.cpp
)

target_compile_features(f8webrtc_gateway_service PRIVATE cxx_std_17)

target_include_directories(f8webrtc_gateway_service
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(websocketpp REQUIRED)
find_package(LibDataChannel REQUIRED)

target_link_libraries(f8webrtc_gateway_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    websocketpp::websocketpp
    LibDataChannel::LibDataChannel
)

target_compile_definitions(f8webrtc_gateway_service PRIVATE _WEBSOCKETPP_CPP11_TYPE_TRAITS_)
