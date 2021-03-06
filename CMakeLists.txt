cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(baryonyx VERSION 0.3.3.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Threads REQUIRED)
add_library(threads INTERFACE IMPORTED)
set_property(TARGET threads PROPERTY
  INTERFACE_LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

find_package(NLOPT)
if (NLOPT_FOUND)
  add_library(nlopt INTERFACE IMPORTED)
  set_property(TARGET nlopt PROPERTY
    INTERFACE_LINK_LIBRARIES ${NLOPT_LIBRARIES})
endif ()

add_definitions(-DFMT_HEADER_ONLY)
include_directories(${CMAKE_SOURCE_DIR}/external/fmt/include)

enable_testing()
add_subdirectory(lib)
add_subdirectory(app)
