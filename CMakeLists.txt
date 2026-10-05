cmake_minimum_required(VERSION 3.20)
project(f8studio VERSION 0.1.0 LANGUAGES CXX)

set(F8_PIXI_CPP_ENV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.pixi/envs/cpp" CACHE PATH "Pixi C++ dependency prefix")
if(EXISTS "${F8_PIXI_CPP_ENV_DIR}")
    # Conan's generated package configs must take precedence over alternate
    # Pixi configs for the same libraries; their imported target names differ.
    list(APPEND CMAKE_PREFIX_PATH "${F8_PIXI_CPP_ENV_DIR}")
    list(PREPEND CMAKE_LIBRARY_PATH "${F8_PIXI_CPP_ENV_DIR}/lib")
    list(PREPEND CMAKE_INCLUDE_PATH "${F8_PIXI_CPP_ENV_DIR}/include")
    if(CMAKE_HOST_WIN32 AND EXISTS "${F8_PIXI_CPP_ENV_DIR}/Library")
        list(APPEND CMAKE_PREFIX_PATH "${F8_PIXI_CPP_ENV_DIR}/Library")
        list(PREPEND CMAKE_LIBRARY_PATH "${F8_PIXI_CPP_ENV_DIR}/Library/lib")
        list(PREPEND CMAKE_INCLUDE_PATH "${F8_PIXI_CPP_ENV_DIR}/Library/include")
    endif()
    if(EXISTS "${F8_PIXI_CPP_ENV_DIR}/bin/pkg-config" AND NOT PKG_CONFIG_EXECUTABLE)
        set(PKG_CONFIG_EXECUTABLE "${F8_PIXI_CPP_ENV_DIR}/bin/pkg-config" CACHE FILEPATH "pkg-config executable")
    endif()
    if(CMAKE_HOST_WIN32)
        set(_f8_pkg_config_separator ";")
    else()
        set(_f8_pkg_config_separator ":")
    endif()
    set(_f8_pixi_pkg_config_paths
        "${F8_PIXI_CPP_ENV_DIR}/lib/pkgconfig"
        "${F8_PIXI_CPP_ENV_DIR}/share/pkgconfig"
    )
    if(CMAKE_HOST_WIN32 AND EXISTS "${F8_PIXI_CPP_ENV_DIR}/Library")
        list(PREPEND _f8_pixi_pkg_config_paths
            "${F8_PIXI_CPP_ENV_DIR}/Library/lib/pkgconfig"
            "${F8_PIXI_CPP_ENV_DIR}/Library/share/pkgconfig"
        )
    endif()
    list(JOIN _f8_pixi_pkg_config_paths "${_f8_pkg_config_separator}" _f8_pixi_pkg_config_path)
    if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
        set(ENV{PKG_CONFIG_PATH} "${_f8_pixi_pkg_config_path}${_f8_pkg_config_separator}$ENV{PKG_CONFIG_PATH}")
    else()
        set(ENV{PKG_CONFIG_PATH} "${_f8_pixi_pkg_config_path}")
    endif()
endif()

if(MSVC)
    # Conan deps here are single-config; force Release for MSVC multi-config generators.
    if(CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
    endif()
endif()

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# package settings
include(GNUInstallDirs)
include(CTest)
enable_testing()

# options
option(BUILD_TESTS "Enable building of tests" OFF)
option(F8_DEPLOY_SERVICE_RUNTIME "Deploy service runtime to build/workspace/runtime/bundles/... after build" ON)
option(F8_DEPLOY_SERVICE_RUNTIME_POST_BUILD "Run deploy as POST_BUILD step" ON)
option(F8_DEPLOY_SERVICE_CLEAN "Clean runtime/bundles/<bundle>/<version>/<platform> before deploy" ON)
option(F8_WITH_ZENOH "Enable Zenoh C/C++ runtime backend when zenohcxx/zenohc are available" ON)

if(NOT TARGET f8_deploy_all_runtime)
    add_custom_target(
        f8_deploy_all_runtime
        COMMENT "Deploy runtime artifacts for all C++ services"
    )
endif()

# Build Output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INCLUDE_CURRENT_DIR ON)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endforeach()

set(CMAKE_DEBUG_POSTFIX "_d")

find_package(Threads REQUIRED)

set(F8_SERVICE_BUNDLES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/workspace/runtime/bundles")

set(F8SDK_BUILD_TESTS ${BUILD_TESTS})
add_subdirectory(sdk)
include(sdk/cmake/F8ServiceRuntime.cmake)
# An empty list builds the SDK only. Each extension also has a standalone entrypoint.
set(F8_EXTENSION_PACKAGES "f8cvkit;f8implayer;f8audiocap;f8screencap;f8cppengine"
    CACHE STRING "Native extension source packages to integrate")
foreach(_f8_extension IN LISTS F8_EXTENSION_PACKAGES)
    if(NOT _f8_extension MATCHES "^f8(cvkit|implayer|audiocap|screencap|cppengine)$")
        message(FATAL_ERROR "Unknown native extension: ${_f8_extension}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extensions/${_f8_extension}/CMakeLists.txt")
        message(FATAL_ERROR "Extension ${_f8_extension} is not checked out. Run git submodule update --init --recursive, or select F8_EXTENSION_PACKAGES explicitly.")
    endif()
    add_subdirectory("extensions/${_f8_extension}")
endforeach()
