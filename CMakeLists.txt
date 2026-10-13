cmake_minimum_required(VERSION 3.18)
project(HORA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HORA_ENABLE_METAL "Enable the Metal GPU backend on macOS if available" ON)
option(HORA_ENABLE_CUDA  "Enable the CUDA GPU backend if a toolkit is available" ON)
option(HORA_BUILD_TESTS  "Build tests" ON)
# `E[] psi` is decided by forward cycle detection over the D-subsumed graph.  -DHORA_LIVENESS_GFP=ON
# selects the backward greatest fixpoint instead, which is the older route: it decides the same
# queries but pays a full `Pre` sweep -- 36.3s against 4.5s on twolock10 q3.  Kept for differential
# testing, not as a supported default.
option(HORA_LIVENESS_GFP "Decide E[] with the backward greatest fixpoint instead of forward cycle detection" OFF)
find_package(OpenMP QUIET)
# Apple Clang ships without an OpenMP runtime, so find_package fails even when
# Homebrew's libomp is installed. Wire it up by hand so the parallel network
# backend (--backend omp) is available out of the box on macOS. Non-Apple
# toolchains (GCC/Clang on Linux) find OpenMP natively above.
if(NOT OpenMP_CXX_FOUND AND APPLE)
  set(_omp_hints /opt/homebrew/opt/libomp /usr/local/opt/libomp)
  find_program(BREW_EXECUTABLE brew)
  if(BREW_EXECUTABLE)
    execute_process(COMMAND ${BREW_EXECUTABLE} --prefix libomp
                    OUTPUT_VARIABLE _omp_brew OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_omp_brew)
      list(PREPEND _omp_hints "${_omp_brew}")
    endif()
  endif()
  foreach(_h ${_omp_hints})
    if(EXISTS "${_h}/lib/libomp.dylib")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_h}/include" CACHE STRING "" FORCE)
      set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
      set(OpenMP_omp_LIBRARY "${_h}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
      find_package(OpenMP QUIET)
      break()
    endif()
  endforeach()
endif()
if(OpenMP_CXX_FOUND)
  message(STATUS "HORA: OpenMP ENABLED — parallel network backend (--backend omp)")
else()
  message(STATUS "HORA: OpenMP not found — network backend runs serially")
endif()

# Optional Metal backend (macOS only, auto-detected).
set(HORA_HAVE_METAL OFF)
if(HORA_ENABLE_METAL AND APPLE)
  find_library(HORA_METAL_FRAMEWORK      Metal)
  find_library(HORA_FOUNDATION_FRAMEWORK Foundation)
  if(HORA_METAL_FRAMEWORK AND HORA_FOUNDATION_FRAMEWORK)
    set(HORA_HAVE_METAL ON)
    message(STATUS "HORA: Metal backend ENABLED (${HORA_METAL_FRAMEWORK})")
  else()
    message(STATUS "HORA: Metal requested but frameworks not found — building without Metal")
  endif()
endif()

# Optional CUDA backend (src/cuda/): the same frontier-batched engine as Metal, on NVIDIA GPUs.
# Only the kernels are nvcc's (frontier_kernels.cu); the host engine is plain C++.
set(HORA_HAVE_CUDA OFF)
if(HORA_ENABLE_CUDA AND NOT APPLE)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80)   # A100; override with -DCMAKE_CUDA_ARCHITECTURES=...
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(HORA_HAVE_CUDA ON)
      set(CMAKE_CUDA_STANDARD 17)
      message(STATUS "HORA: CUDA backend ENABLED (${CMAKE_CUDA_COMPILER}, arch ${CMAKE_CUDA_ARCHITECTURES})")
    endif()
  endif()
endif()

# ---- HDBM: the difference-bound-matrix zone library, now standalone (its own repo). -----
# Prefer an in-tree checkout (HDBM/); if it is absent, fetch it. HDBM exports target `hdbm`
# and owns the zone algebra the CPU and Metal paths share.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/HDBM/CMakeLists.txt)
  message(STATUS "HORA: using in-tree HDBM (${CMAKE_CURRENT_SOURCE_DIR}/HDBM)")
  add_subdirectory(HDBM)
else()
  message(STATUS "HORA: HDBM not found in tree — fetching it via FetchContent")
  include(FetchContent)
  FetchContent_Declare(hdbm
    GIT_REPOSITORY https://github.com/horaverify/HDBM.git   # FIXME: placeholder — real repo TBD
    GIT_TAG        main)
  FetchContent_MakeAvailable(hdbm)
endif()

# The engine's translation units, by area. A glob per directory rather than one recursive glob,
# so that adding a directory is a deliberate edit here and the layout stays visible.
# The engine's translation units, one glob per area, so that adding an area is a deliberate
# edit here and the layout of src/ stays visible from the build file.
file(GLOB HORA_CORE_SRC CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/analysis/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/liveness/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/query/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/reachability/*.cpp)
add_library(hora_core ${HORA_CORE_SRC})
target_include_directories(hora_core
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})   # so a TU can say "src/<area>/<internal>.hpp"
target_compile_features(hora_core PUBLIC cxx_std_17)
target_link_libraries(hora_core PUBLIC hdbm)   # DBM zone algebra + its headers
if(OpenMP_CXX_FOUND)
  target_link_libraries(hora_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(hora_core PUBLIC HORA_HAVE_OPENMP=1)
endif()
if(HORA_HAVE_METAL)
  file(GLOB HORA_METAL_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.mm)
  if(HORA_METAL_SRC)
    target_sources(hora_core PRIVATE ${HORA_METAL_SRC})
    set_source_files_properties(${HORA_METAL_SRC} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(hora_core PUBLIC
      ${HORA_METAL_FRAMEWORK} ${HORA_FOUNDATION_FRAMEWORK})
  endif()
  target_compile_definitions(hora_core PUBLIC HORA_HAVE_METAL=1)
endif()
if(HORA_HAVE_CUDA)
  file(GLOB HORA_CUDA_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp)
  target_sources(hora_core PRIVATE ${HORA_CUDA_SRC})
  target_link_libraries(hora_core PUBLIC CUDA::cudart_static)
  target_include_directories(hora_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(hora_core PUBLIC HORA_HAVE_CUDA=1)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/frontier_kernels.cu
                              PROPERTIES COMPILE_OPTIONS "-lineinfo")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(hora_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
endif()

if(HORA_LIVENESS_GFP)
  target_compile_definitions(hora_core PUBLIC HORA_LIVENESS_GFP=1)
endif()

add_executable(hora ${CMAKE_CURRENT_SOURCE_DIR}/app/main.cpp)
target_link_libraries(hora PRIVATE hora_core)

if(HORA_BUILD_TESTS)
  enable_testing()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
  endif()
endif()
