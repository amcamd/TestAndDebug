# Header

set( TensileClient_SOLUTIONS
  ${CMAKE_SOURCE_DIR}/Solutions.h
  ${CMAKE_SOURCE_DIR}/Solutions.cpp
  )
set( TensileClient_KERNELS
  ${CMAKE_SOURCE_DIR}/Kernels.h
  ${CMAKE_SOURCE_DIR}/Kernels.cpp
  )
set( TensileClient_SOURCE
  ${CMAKE_SOURCE_DIR}/Client.cpp
  ${CMAKE_SOURCE_DIR}/Client.h
  ${CMAKE_SOURCE_DIR}/CMakeLists.txt
  ${CMAKE_SOURCE_DIR}/MathTemplates.cpp
  ${CMAKE_SOURCE_DIR}/MathTemplates.h
  ${CMAKE_SOURCE_DIR}/TensileTypes.h
  ${CMAKE_SOURCE_DIR}/ReferenceCPU.h
  ${CMAKE_SOURCE_DIR}/SolutionHelper.cpp
  ${CMAKE_SOURCE_DIR}/SolutionHelper.h
  ${CMAKE_SOURCE_DIR}/Tools.cpp
  ${CMAKE_SOURCE_DIR}/Tools.h
  )

