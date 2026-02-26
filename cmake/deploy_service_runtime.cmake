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

if(DEFINED clean_dest)
  string(STRIP "${clean_dest}" clean_dest)
  string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" clean_dest "${clean_dest}")
endif()

if(DEFINED clean_dest AND NOT "${clean_dest}" STREQUAL "" AND clean_dest)
  file(GLOB _old_files
    "${dest_dir}/*.dll"
    "${dest_dir}/*.exe"
    "${dest_dir}/*.pdb"
    "${dest_dir}/*.so"
    "${dest_dir}/*.so.*"
    "${dest_dir}/*.dylib"
    "${dest_dir}/*.dylib.*"
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

# Include runtime search env vars. Using PATH on Linux tends to pull in many
# unrelated system packages and slows dependency resolution.
if(WIN32)
  if(DEFINED ENV{PATH} AND NOT "$ENV{PATH}" STREQUAL "")
    string(REPLACE ";" ";" _path_list "$ENV{PATH}")
    list(APPEND _search_dirs ${_path_list})
  endif()
elseif(APPLE)
  if(DEFINED ENV{DYLD_LIBRARY_PATH} AND NOT "$ENV{DYLD_LIBRARY_PATH}" STREQUAL "")
    string(REPLACE ":" ";" _dyld_list "$ENV{DYLD_LIBRARY_PATH}")
    list(APPEND _search_dirs ${_dyld_list})
  endif()
else()
  if(DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    string(REPLACE ":" ";" _ld_path_list "$ENV{LD_LIBRARY_PATH}")
    list(APPEND _search_dirs ${_ld_path_list})
  endif()
endif()
list(REMOVE_DUPLICATES _search_dirs)

set(_resolved "")
set(_unresolved "")

# Avoid copying OS/system runtime libs.
set(_pre_exclude_regexes
  [[^api-ms-win-.*]]
  [[^ext-ms-win-.*]]
  [[.*[/\\]Windows[/\\].*]]
  [[.*[/\\]System32[/\\].*]]
  [[.*[/\\]WinSxS[/\\].*]]
  [[.*[/\\]system32[/\\].*]]
  [[.*[/\\]winsxs[/\\].*]]
)
set(_post_exclude_regexes "")
if(UNIX AND NOT APPLE)
  # Skip Linux core runtime libs; they should come from the host system.
  list(APPEND _pre_exclude_regexes
    [[^linux-vdso\.so.*]]
    [[^ld-linux-.*\.so.*]]
    [[^lib(c|m|dl|pthread|rt|util|resolv|nsl|anl|crypt)\.so.*]]
    [[^libnss_.*\.so.*]]
    [[^libBrokenLocale\.so.*]]
    [[^libthread_db\.so.*]]
  )
  # Do not bundle host system library directories.
  list(APPEND _post_exclude_regexes
    [[^/lib/.*]]
    [[^/lib32/.*]]
    [[^/lib64/.*]]
    [[^/usr/lib/.*]]
    [[^/usr/lib32/.*]]
    [[^/usr/lib64/.*]]
    [[^/usr/local/lib/.*]]
  )
endif()

file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${exe}"
  DIRECTORIES ${_search_dirs}
  RESOLVED_DEPENDENCIES_VAR _resolved
  UNRESOLVED_DEPENDENCIES_VAR _unresolved
  PRE_EXCLUDE_REGEXES ${_pre_exclude_regexes}
  POST_EXCLUDE_REGEXES ${_post_exclude_regexes}
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
  if(UNIX AND NOT APPLE)
    if(_dep_norm MATCHES "^/lib/" OR _dep_norm MATCHES "^/lib32/" OR _dep_norm MATCHES "^/lib64/" OR
       _dep_norm MATCHES "^/usr/lib/" OR _dep_norm MATCHES "^/usr/lib32/" OR _dep_norm MATCHES "^/usr/lib64/" OR
       _dep_norm MATCHES "^/usr/local/lib/")
      continue()
    endif()
  endif()

  get_filename_component(_name "${_dep}" NAME)
  if(UNIX AND NOT APPLE)
    string(TOLOWER "${_name}" _name_lower)
    if(_name_lower MATCHES [[^(linux-vdso\.so.*|ld-linux-.*\.so.*|lib(c|m|dl|pthread|rt|util|resolv|nsl|anl|crypt)\.so.*|libnss_.*\.so.*|libbrokenlocale\.so.*|libthread_db\.so.*)$]])
      continue()
    endif()
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dep}" "${dest_dir}/${_name}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endforeach()

if(DEFINED extra_files AND NOT "${extra_files}" STREQUAL "")
  string(STRIP "${extra_files}" extra_files)
  string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" extra_files "${extra_files}")
  string(REPLACE "|" ";" extra_files "${extra_files}")
  foreach(_extra IN LISTS extra_files)
    if(NOT EXISTS "${_extra}")
      message(WARNING "deploy_service_runtime.cmake: extra file does not exist, skipping: ${_extra}")
      continue()
    endif()
    get_filename_component(_extra_name "${_extra}" NAME)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_extra}" "${dest_dir}/${_extra_name}"
      COMMAND_ERROR_IS_FATAL ANY
    )
  endforeach()
endif()

if(_unresolved)
  list(JOIN _unresolved "\n  " _u)
  message(WARNING "deploy_service_runtime.cmake: unresolved runtime deps for ${exe}:\n  ${_u}")
endif()
