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
find_package(OpenCV REQUIRED)
find_package(openh264 REQUIRED)
find_package(libvpx REQUIRED)
find_package(gstreamer REQUIRED)
find_package(glib REQUIRED)
find_package(gst-plugins-bad QUIET)
find_package(gst-plugins-base QUIET)
find_package(gst-plugins-good QUIET)

find_path(GSTREAMER_INCLUDE_DIR NAMES gst/gst.h PATH_SUFFIXES gstreamer-1.0)
if (GSTREAMER_INCLUDE_DIR)
  target_include_directories(f8webrtc_gateway_service PRIVATE ${GSTREAMER_INCLUDE_DIR})
endif()

# Conan's GLib CMake package may not declare include/lib info on Windows reliably.
find_path(GLIB_INCLUDE_DIR NAMES glib.h PATH_SUFFIXES glib-2.0)
find_path(GLIBCONFIG_INCLUDE_DIR NAMES glibconfig.h PATH_SUFFIXES lib/glib-2.0/include)
if (GLIB_INCLUDE_DIR)
  target_include_directories(f8webrtc_gateway_service PRIVATE ${GLIB_INCLUDE_DIR})
endif()
if (GLIBCONFIG_INCLUDE_DIR)
  target_include_directories(f8webrtc_gateway_service PRIVATE ${GLIBCONFIG_INCLUDE_DIR})
endif()

find_library(GLIB_LIB NAMES glib-2.0 PATH_SUFFIXES lib)
find_library(GOBJECT_LIB NAMES gobject-2.0 PATH_SUFFIXES lib)
find_library(GIO_LIB NAMES gio-2.0 PATH_SUFFIXES lib)
find_library(GMODULE_LIB NAMES gmodule-2.0 PATH_SUFFIXES lib)
find_library(GTHREAD_LIB NAMES gthread-2.0 PATH_SUFFIXES lib)
if (GLIB_LIB)
  target_link_libraries(f8webrtc_gateway_service PRIVATE ${GLIB_LIB})
endif()
if (GOBJECT_LIB)
  target_link_libraries(f8webrtc_gateway_service PRIVATE ${GOBJECT_LIB})
endif()
if (GIO_LIB)
  target_link_libraries(f8webrtc_gateway_service PRIVATE ${GIO_LIB})
endif()
if (GMODULE_LIB)
  target_link_libraries(f8webrtc_gateway_service PRIVATE ${GMODULE_LIB})
endif()
if (GTHREAD_LIB)
  target_link_libraries(f8webrtc_gateway_service PRIVATE ${GTHREAD_LIB})
endif()

# Conan's gstreamer CMake package may not declare import libs/includes on Windows reliably.
# Fall back to locating the import library directly from the Conan package prefix.
find_library(GSTREAMER_CORE_LIB NAMES gstreamer-1.0 PATH_SUFFIXES lib)
if (GSTREAMER_CORE_LIB)
  target_link_libraries(f8webrtc_gateway_service PRIVATE ${GSTREAMER_CORE_LIB})
endif()
find_path(OPENH264_INCLUDE_DIR NAMES wels/codec_api.h)
if (OPENH264_INCLUDE_DIR)
  target_include_directories(f8webrtc_gateway_service PRIVATE ${OPENH264_INCLUDE_DIR})
endif()
find_path(LIBVPX_INCLUDE_DIR NAMES vpx/vpx_decoder.h)
if (LIBVPX_INCLUDE_DIR)
  target_include_directories(f8webrtc_gateway_service PRIVATE ${LIBVPX_INCLUDE_DIR})
endif()

target_link_libraries(f8webrtc_gateway_service
  PRIVATE
    f8cppsdk
    cxxopts::cxxopts
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    websocketpp::websocketpp
    LibDataChannel::LibDataChannel
    opencv::opencv
    openh264::openh264
    libvpx::libvpx
    gstreamer::gstreamer
    glib::glib
)

target_compile_definitions(f8webrtc_gateway_service PRIVATE _WEBSOCKETPP_CPP11_TYPE_TRAITS_)

if(gst-plugins-bad_FOUND AND gst-plugins-base_FOUND)
  target_compile_definitions(f8webrtc_gateway_service PRIVATE F8_WITH_GST_WEBRTC=1 GST_USE_UNSTABLE_API)
  target_link_libraries(f8webrtc_gateway_service PRIVATE gst-plugins-bad::gst-plugins-bad gst-plugins-base::gst-plugins-base)
  if(gst-plugins-good_FOUND)
    target_link_libraries(f8webrtc_gateway_service PRIVATE gst-plugins-good::gst-plugins-good)
  endif()
endif()

add_executable(f8gst_probe
  src/gst_probe.cpp
)

target_compile_features(f8gst_probe PRIVATE cxx_std_17)

target_link_libraries(f8gst_probe
  PRIVATE
    gstreamer::gstreamer
    glib::glib
)

if (GSTREAMER_INCLUDE_DIR)
  target_include_directories(f8gst_probe PRIVATE ${GSTREAMER_INCLUDE_DIR})
endif()

if (GLIB_INCLUDE_DIR)
  target_include_directories(f8gst_probe PRIVATE ${GLIB_INCLUDE_DIR})
endif()
if (GLIBCONFIG_INCLUDE_DIR)
  target_include_directories(f8gst_probe PRIVATE ${GLIBCONFIG_INCLUDE_DIR})
endif()

if (GLIB_LIB)
  target_link_libraries(f8gst_probe PRIVATE ${GLIB_LIB})
endif()
if (GOBJECT_LIB)
  target_link_libraries(f8gst_probe PRIVATE ${GOBJECT_LIB})
endif()

if (GSTREAMER_CORE_LIB)
  target_link_libraries(f8gst_probe PRIVATE ${GSTREAMER_CORE_LIB})
endif()
