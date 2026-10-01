include_guard(GLOBAL)
include(GNUInstallDirs)
include(CTest)
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
endif()
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()
if(NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()
foreach(_f8_configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${_f8_configuration}" _f8_configuration)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_f8_configuration} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_f8_configuration} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_f8_configuration} "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
endforeach()
set(_F8_SERVICE_RUNTIME_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")
if(NOT DEFINED F8_SERVICE_BUNDLES_ROOT)
    set(F8_SERVICE_BUNDLES_ROOT "${CMAKE_BINARY_DIR}/runtime/bundles" CACHE PATH "Service runtime output directory")
endif()
option(F8_DEPLOY_SERVICE_RUNTIME "Deploy service runtime after build" ON)
option(F8_DEPLOY_SERVICE_RUNTIME_POST_BUILD "Deploy automatically after linking" ON)
option(F8_DEPLOY_SERVICE_CLEAN "Clean service runtime before deployment" ON)

function(f8_deploy_service_runtime target service_rel_dir)
    if(NOT TARGET ${target})
        message(WARNING "f8_deploy_service_runtime: target not found: ${target}")
        return()
    endif()

    # Optional args:
    # - "NO_CLEAN" to allow multiple executables to share one runtime directory.
    # - "EXTRA_FILES" followed by one or more file paths to copy into deploy dir.
    set(_extra_files "")
    set(_clean_dest "$<IF:$<BOOL:${F8_DEPLOY_SERVICE_CLEAN}>,1,0>")
    set(_arg_index 2)
    while(_arg_index LESS ARGC)
        set(_arg "${ARGV${_arg_index}}")
        if("${_arg}" STREQUAL "NO_CLEAN")
            set(_clean_dest "0")
        elseif("${_arg}" STREQUAL "EXTRA_FILES")
            math(EXPR _arg_index "${_arg_index} + 1")
            while(_arg_index LESS ARGC)
                set(_next "${ARGV${_arg_index}}")
                if("${_next}" STREQUAL "NO_CLEAN" OR "${_next}" STREQUAL "EXTRA_FILES")
                    math(EXPR _arg_index "${_arg_index} - 1")
                    break()
                endif()
                list(APPEND _extra_files "${_next}")
                math(EXPR _arg_index "${_arg_index} + 1")
            endwhile()
        endif()
        math(EXPR _arg_index "${_arg_index} + 1")
    endwhile()

    if(WIN32)
        set(_platform_dir "win")
    elseif(APPLE)
        set(_platform_dir "mac")
    else()
        set(_platform_dir "linux")
    endif()

    set(_dest_dir "${F8_SERVICE_BUNDLES_ROOT}/${service_rel_dir}/0.0.1/${_platform_dir}")
    set(_bin_dir "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
    set(_lib_dir "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

    set(_deploy_cmd
        "${CMAKE_COMMAND}"
        -Dexe="$<TARGET_FILE:${target}>"
        -Ddest_dir="${_dest_dir}"
        -Dbin_dir="${_bin_dir}"
        -Dlib_dir="${_lib_dir}"
        -Dclean_dest="${_clean_dest}"
        -P "${_F8_SERVICE_RUNTIME_MODULE_DIR}/deploy_service_runtime.cmake"
    )
    if(_extra_files)
        # Use '|' as transport separator to avoid ';' list splitting in Visual Studio custom commands.
        list(APPEND _deploy_cmd -Dextra_files="$<JOIN:${_extra_files},|>")
    endif()

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
    if(TARGET f8_deploy_all_runtime)
        add_dependencies(f8_deploy_all_runtime ${_deploy_target})
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
