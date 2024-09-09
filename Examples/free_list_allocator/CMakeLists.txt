cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/amdclang++")
#set(ROCBLAS_INSTALL "/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS-internal/build_tensile/release/rocblas-install")
#set(ROCBLAS_INTERNAL "/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS-internal/library/src/")

project ( free_list_allocator LANGUAGES CXX )

#if( NOT (CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hcc$"))
#if( NOT (CMAKE_CXX_COMPILER MATCHES ".*/amdclang++$"))
#    message(STATUS "-----------------------------------------------------------")
#    message(STATUS "CMAKE_CXX_COMPILER = " ${CMAKE_CXX_COMPILER})
#    message(STATUS "try setting environment variable CXX to /opt/rocm/bin/amdclang++")
#    message(STATUS "for example, run:   CXX=/opt/rocm/bin/amdclang++ cmake ..")
#    message(STATUS "-----------------------------------------------------------")
#endif()

#SET(CXX_BUILTIN "")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_BUILTIN} -std=c++17")

SET(SOURCES main.cpp)

#include_directories(${ROCBLAS_INSTALL}/include/rocblas ${ROCBLAS_INTERNAL}/include)
#link_directories(${ROCBLAS_INSTALL}/lib)

add_executable(free_list_allocator main.cpp)

#target_link_libraries(scal_test librocblas.so)
