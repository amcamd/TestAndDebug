cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")

project ( bf16_test LANGUAGES CXX )

#if( NOT (CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hcc$"))
if( NOT (CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(STATUS "-----------------------------------------------------------")
    message(STATUS "CMAKE_CXX_COMPILER = " ${CMAKE_CXX_COMPILER})
    message(STATUS "try setting environment variable CXX to /opt/rocm/bin/hipcc")
    message(STATUS "for example, run:   CXX=/opt/rocm/bin/hipcc cmake ..")
    message(STATUS "-----------------------------------------------------------")
endif()

SET(CXX_CXX11 "-std=c++11")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_BUILTIN}")

add_executable(bf16_test main.cpp)

