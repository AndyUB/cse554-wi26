cmake_minimum_required(VERSION 3.10)

project(CSE554 LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Default arch if user didn't provide one (RTX 6000 = 75)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(
  copy
  main.cu
  copy_first_column.cu
)

# Ensure this target is built for the requested arch(es)
set_property(TARGET copy PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Common (safe) CUDA compile flags
target_compile_options(copy PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing>
)

# Add Hopper "90a" codegen only if building for 90/90a.
if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "(^|;)(90|90a)($|;)")
  target_compile_options(copy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]>
  )
endif()


