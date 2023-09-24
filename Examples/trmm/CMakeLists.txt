cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(ROCBLAS_INSTALL "/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS-internal/build_tensile/release/rocblas-install")
set(ROCBLAS_INTERNAL "/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS-internal/library/src/")

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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_BUILTIN} -std=c++14")

SET(SOURCES main.cpp)

include_directories(${ROCBLAS_INSTALL}/include/rocblas ${ROCBLAS_INTERNAL}/include)
link_directories(${ROCBLAS_INSTALL}/lib)

add_executable(trmm_test main.cpp)

target_link_libraries(trmm_test librocblas.so)
