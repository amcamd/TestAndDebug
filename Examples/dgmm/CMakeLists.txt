cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
#set(ROCBLAS_INSTALL "/opt/rocm-3.3.0")
set(ROCBLAS_INSTALL "/opt/rocm")

project ( i8_test LANGUAGES CXX )

#if( NOT (CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hcc$"))
if( NOT (CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(STATUS "-----------------------------------------------------------")
    message(STATUS "CMAKE_CXX_COMPILER = " ${CMAKE_CXX_COMPILER})
    message(STATUS "try setting environment variable CXX to /opt/rocm/bin/hipcc")
    message(STATUS "for example, run:   CXX=/opt/rocm/bin/hipcc cmake ..")
    message(STATUS "-----------------------------------------------------------")
endif()

SET(CXX_BUILTIN "")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_BUILTIN}")

SET(SOURCES main.cpp)

include_directories(${ROCBLAS_INSTALL}/include)
link_directories(${ROCBLAS_INSTALL}/lib)

add_executable(dgmm_test main.cpp)

target_link_libraries(dgmm_test librocblas.so)
