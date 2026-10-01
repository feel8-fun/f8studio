cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED source_root OR "${source_root}" STREQUAL "")
  message(FATAL_ERROR "deploy_cpython_runtime.cmake: missing -Dsource_root=...")
endif()
if(NOT DEFINED dest_dir OR "${dest_dir}" STREQUAL "")
  message(FATAL_ERROR "deploy_cpython_runtime.cmake: missing -Ddest_dir=...")
endif()

string(STRIP "${source_root}" source_root)
string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" source_root "${source_root}")
string(STRIP "${dest_dir}" dest_dir)
string(REGEX REPLACE "^\\\"(.*)\\\"$" "\\1" dest_dir "${dest_dir}")

if(NOT EXISTS "${source_root}/Lib/os.py")
  message(FATAL_ERROR "deploy_cpython_runtime.cmake: source_root does not look like a CPython runtime: ${source_root}")
endif()

file(MAKE_DIRECTORY "${dest_dir}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_directory "${source_root}/Lib" "${dest_dir}/Lib"
  COMMAND_ERROR_IS_FATAL ANY
)

if(EXISTS "${source_root}/DLLs")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${source_root}/DLLs" "${dest_dir}/DLLs"
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

file(GLOB _python_dlls "${source_root}/python*.dll")
foreach(_dll IN LISTS _python_dlls)
  get_filename_component(_name "${_dll}" NAME)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dll}" "${dest_dir}/${_name}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endforeach()

if(EXISTS "${source_root}/Library/bin")
  file(GLOB _library_dlls "${source_root}/Library/bin/*.dll")
  foreach(_dll IN LISTS _library_dlls)
    get_filename_component(_name "${_dll}" NAME)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dll}" "${dest_dir}/${_name}"
      COMMAND_ERROR_IS_FATAL ANY
    )
  endforeach()
endif()
