cmake_minimum_required(VERSION 3.20)
project(f8studio VERSION 0.1.0 LANGUAGES CXX)

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

add_subdirectory(packages/f8cppsdk)
add_subdirectory(packages/f8implayer)
add_subdirectory(packages/f8audiocap)

if (BUILD_TESTS)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(WARNING "BUILD_TESTS=ON but ./tests is missing; skipping")
    endif()
endif()

# # Packaging
# include(CPack)
# set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
# set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")

# # Install configuration
# install(EXPORT f8playerTargets
#     FILE f8playerTargets.cmake
#     NAMESPACE f8player::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/f8player
# )
