cmake_minimum_required(VERSION 3.18)
project(gfp_proto LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(REPO /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC)

# HDBM comes from the repo, unmodified.
add_subdirectory(${REPO}/HDBM ${CMAKE_BINARY_DIR}/hdbm)

# Every core source EXCEPT netreach.cpp, which is replaced by our instrumented copy.
# The copy keeps the same external symbols, so nothing else has to change.
file(GLOB PROTO_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(gfp_proto ${PROTO_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/main_gfp.cpp)
target_include_directories(gfp_proto PRIVATE ${REPO}/include)
target_link_libraries(gfp_proto PRIVATE hdbm)
target_compile_options(gfp_proto PRIVATE -Wno-unused-parameter -Wno-unused-variable)
