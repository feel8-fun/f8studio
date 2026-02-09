cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED exe)
  message(FATAL_ERROR "deploy_service_runtime.cmake: missing -Dexe=...")
endif()
string(STRIP "${exe}" exe)
string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" exe "${exe}")
if(NOT EXISTS "${exe}")
  message(FATAL_ERROR "deploy_service_runtime.cmake: exe does not exist: ${exe}")
endif()
if(NOT DEFINED dest_dir)
  message(FATAL_ERROR "deploy_service_runtime.cmake: missing -Ddest_dir=...")
endif()
string(STRIP "${dest_dir}" dest_dir)
string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" dest_dir "${dest_dir}")

set(extra_search_dirs "")
if(DEFINED bin_dir AND NOT "${bin_dir}" STREQUAL "")
  string(STRIP "${bin_dir}" bin_dir)
  string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" bin_dir "${bin_dir}")
  list(APPEND extra_search_dirs "${bin_dir}")
endif()
if(DEFINED lib_dir AND NOT "${lib_dir}" STREQUAL "")
  string(STRIP "${lib_dir}" lib_dir)
  string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" lib_dir "${lib_dir}")
  list(APPEND extra_search_dirs "${lib_dir}")
endif()

file(MAKE_DIRECTORY "${dest_dir}")

if(DEFINED clean_dest AND clean_dest)
  file(GLOB _old_files
    "${dest_dir}/*.dll"
    "${dest_dir}/*.exe"
    "${dest_dir}/*.pdb"
    "${dest_dir}/*.so"
    "${dest_dir}/*.dylib"
  )
  if(_old_files)
    file(REMOVE ${_old_files})
  endif()
endif()

get_filename_component(_exe_name "${exe}" NAME)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${exe}" "${dest_dir}/${_exe_name}"
  COMMAND_ERROR_IS_FATAL ANY
)

get_filename_component(_exe_dir "${exe}" DIRECTORY)
set(_search_dirs "${_exe_dir}")
list(APPEND _search_dirs ${extra_search_dirs})

# Also include PATH (useful for dev builds where dependencies live in a toolchain/env).
if(DEFINED ENV{PATH} AND NOT "$ENV{PATH}" STREQUAL "")
  if(WIN32)
    string(REPLACE ";" ";" _path_list "$ENV{PATH}")
  else()
    string(REPLACE ":" ";" _path_list "$ENV{PATH}")
  endif()
  list(APPEND _search_dirs ${_path_list})
endif()
list(REMOVE_DUPLICATES _search_dirs)

set(_resolved "")
set(_unresolved "")

# Avoid copying OS/system runtime libs.
set(_exclude_regexes
  [[^api-ms-win-.*]]
  [[^ext-ms-win-.*]]
  [[.*[/\\]Windows[/\\].*]]
  [[.*[/\\]System32[/\\].*]]
  [[.*[/\\]WinSxS[/\\].*]]
  [[.*[/\\]system32[/\\].*]]
  [[.*[/\\]winsxs[/\\].*]]
)

file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${exe}"
  DIRECTORIES ${_search_dirs}
  RESOLVED_DEPENDENCIES_VAR _resolved
  UNRESOLVED_DEPENDENCIES_VAR _unresolved
  PRE_EXCLUDE_REGEXES ${_exclude_regexes}
)

foreach(_dep IN LISTS _resolved)
  if(NOT EXISTS "${_dep}")
    continue()
  endif()

  # Never bundle Windows system DLLs.
  string(TOLOWER "${_dep}" _dep_lower)
  string(REPLACE "\\" "/" _dep_norm "${_dep_lower}")
  if(_dep_norm MATCHES "/windows/" OR _dep_norm MATCHES "/system32/" OR _dep_norm MATCHES "/winsxs/")
    continue()
  endif()

  get_filename_component(_name "${_dep}" NAME)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dep}" "${dest_dir}/${_name}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endforeach()

if(_unresolved)
  list(JOIN _unresolved "\n  " _u)
  message(WARNING "deploy_service_runtime.cmake: unresolved runtime deps for ${exe}:\n  ${_u}")
endif()
