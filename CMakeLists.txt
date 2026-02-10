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
option(F8_DEPLOY_SERVICE_RUNTIME "Deploy service runtime to services/... after build" ON)
option(F8_DEPLOY_SERVICE_RUNTIME_POST_BUILD "Run deploy as POST_BUILD step" ON)
option(F8_DEPLOY_SERVICE_CLEAN "Clean services/<svc>/<platform> before deploy" ON)

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

set(F8_SERVICES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/services")

function(f8_deploy_service_runtime target service_rel_dir)
    if(NOT TARGET ${target})
        message(WARNING "f8_deploy_service_runtime: target not found: ${target}")
        return()
    endif()

    if(WIN32)
        set(_platform_dir "win")
    elseif(APPLE)
        set(_platform_dir "mac")
    else()
        set(_platform_dir "linux")
    endif()

    set(_dest_dir "${F8_SERVICES_ROOT}/${service_rel_dir}/${_platform_dir}")
    set(_bin_dir "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
    set(_lib_dir "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

    set(_deploy_cmd
        "${CMAKE_COMMAND}"
        -Dexe="$<TARGET_FILE:${target}>"
        -Ddest_dir="${_dest_dir}"
        -Dbin_dir="${_bin_dir}"
        -Dlib_dir="${_lib_dir}"
        -Dclean_dest="$<IF:$<BOOL:${F8_DEPLOY_SERVICE_CLEAN}>,1,0>"
        -P "${PROJECT_SOURCE_DIR}/cmake/deploy_service_runtime.cmake"
    )

    # Always provide a manual deploy target, so developers can disable POST_BUILD
    # and run deploy explicitly when needed.
    set(_deploy_target "${target}_deploy_runtime")
    if(NOT TARGET ${_deploy_target})
        add_custom_target(
            ${_deploy_target}
            COMMAND ${_deploy_cmd}
            DEPENDS ${target}
            VERBATIM
            COMMENT "Deploy ${target} runtime to ${_dest_dir}"
        )
    endif()

    if(F8_DEPLOY_SERVICE_RUNTIME AND F8_DEPLOY_SERVICE_RUNTIME_POST_BUILD)
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${_deploy_cmd}
            VERBATIM
            COMMENT "Deploy ${target} runtime to ${_dest_dir}"
        )
    endif()
endfunction()

add_subdirectory(packages/f8cppsdk)
add_subdirectory(packages/f8implayer)
add_subdirectory(packages/f8audiocap)
add_subdirectory(packages/f8screencap)
add_subdirectory(packages/f8sdk_demo)

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
