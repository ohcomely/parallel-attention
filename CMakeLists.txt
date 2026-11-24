cmake_minimum_required(VERSION 3.20)
project(parallel_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ATTN_GPU "Build the CUDA/cuBLAS path (not implemented yet)" OFF)
option(ATTN_OPENMP "Thread the softmax kernels with OpenMP (if found)" ON)
option(ATTN_NATIVE "Compile for the build machine's CPU (-march=native); build on a compute node" OFF)
set(ATTN_MPIEXEC_EXTRA "--oversubscribe" CACHE STRING "Extra mpiexec flags for tests")
set(ATTN_TEST_RANKS 1 2 4 8 CACHE STRING "Rank counts used by ctest")

find_package(MPI REQUIRED COMPONENTS CXX)

if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  set(BLAS_LIBS ${ACCELERATE})
  set(BLAS_DEFS ATTN_ACCELERATE ACCELERATE_NEW_LAPACK)
  set(BLAS_NAME Accelerate)
else()
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)
  set(BLAS_LIBS ${BLAS_LIBRARIES})
  set(BLAS_NAME OpenBLAS)
  # FindBLAS gives no include directory; look for cblas.h next to the library (e.g. <prefix>/lib64 -> <prefix>/include).
  list(GET BLAS_LIBRARIES 0 BLAS_FIRST_LIB)
  get_filename_component(BLAS_LIB_DIR "${BLAS_FIRST_LIB}" DIRECTORY)
  find_path(CBLAS_INCLUDE_DIR cblas.h HINTS "${BLAS_LIB_DIR}/../include" PATH_SUFFIXES openblas REQUIRED)
endif()

# Library: distributed attention (include/attn, src).
add_library(attn
  src/attention.cpp
  src/registry.cpp
  src/costmodel.cpp
  src/comm.cpp
  src/grid.cpp
  src/layout.cpp
  src/timers.cpp
  src/kernels_cpu.cpp
  src/algs/attn3d.cpp
  src/algs/split.cpp
  src/algs/ring.cpp
  src/algs/softmax_v.cpp)
target_include_directories(attn PUBLIC include PRIVATE src)
if(CBLAS_INCLUDE_DIR)
  target_include_directories(attn PRIVATE ${CBLAS_INCLUDE_DIR})
endif()
target_link_libraries(attn PUBLIC MPI::MPI_CXX ${BLAS_LIBS})
target_compile_definitions(attn PRIVATE ${BLAS_DEFS} ATTN_BLAS_NAME="${BLAS_NAME}")
target_compile_options(attn PRIVATE -Wall -Wextra)
# Softmax kernels: `omp simd` loops vectorise with -fopenmp-simd alone; OpenMP threads split rows.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fopenmp-simd ATTN_HAS_OPENMP_SIMD)
if(ATTN_HAS_OPENMP_SIMD)
  target_compile_options(attn PRIVATE -fopenmp-simd)
endif()
if(ATTN_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(attn PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found: softmax kernels run on one thread")
  endif()
endif()
if(ATTN_NATIVE)
  target_compile_options(attn PRIVATE -march=native)
endif()

# Benchmark harness for the paper: generated inputs, fp64 reference, measurement protocol, CSV output.
add_library(attn_bench
  bench/inputs.cpp
  bench/reference.cpp
  bench/stats.cpp
  bench/harness.cpp
  bench/report.cpp)
target_include_directories(attn_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(attn_bench PUBLIC attn)
target_compile_options(attn_bench PRIVATE -Wall -Wextra)

add_executable(attnbench bench/main.cpp)
target_link_libraries(attnbench PRIVATE attn_bench)

enable_testing()
add_executable(test_costmodel tests/test_costmodel.cpp)
target_link_libraries(test_costmodel PRIVATE attn_bench)
add_test(NAME test_costmodel COMMAND test_costmodel)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME test_tools COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_tools.py)
endif()

foreach(test test_correctness test_words test_timers)
  add_executable(${test} tests/${test}.cpp)
  target_link_libraries(${test} PRIVATE attn_bench)
  foreach(np ${ATTN_TEST_RANKS})
    add_test(NAME ${test}_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${ATTN_MPIEXEC_EXTRA} $<TARGET_FILE:${test}>)
    # Small problems, several ranks per core set: threaded BLAS would only oversubscribe the cores.
    set_tests_properties(${test}_np${np} PROPERTIES
                         ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1")
  endforeach()
endforeach()
