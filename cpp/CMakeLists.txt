#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../cmake/rapids_config.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUMLPRIMS_MG)

project(CUMLPRIMS_MG VERSION "${RAPIDS_VERSION}" LANGUAGES CXX CUDA)

# Write the version header
rapids_cmake_write_version_file(include/cumlprims/version_config.hpp)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###################################################################################################
# - Build Options  ---------------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build libcumlprims shared libraries" ON)
option(BUILD_CUMLPRIMS_LIBRARY "Build libcumlprims library" ON)
option(BUILD_TESTS "Build the cumlprims OPG test executable" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(CUDA_STATIC_MATH_LIBRARIES "Statically link the CUDA math libraries" OFF)
option(CUMLPRIMS_MG_RAFT_CLONE_ON_PIN "Explicitly clone RAFT branch when pinned to non-feature branch" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(NVTX "Enable nvtx markers" OFF)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." OFF)

if(BUILD_TESTS)
  message(STATUS "CUMLPRIMS_MG: Setting BUILD_TESTS to OFF because they don't compile")
  set(BUILD_TESTS OFF)
endif()

message(VERBOSE "CUMLPRIMS_MG: Building libcumlprims shared library: ${BUILD_CUMLPRIMS_LIBRARY}")
message(VERBOSE "CUMLPRIMS_MG: Building tests: ${BUILD_TESTS}")
message(VERBOSE "CUMLPRIMS_MG: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "CUMLPRIMS_MG: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "CUMLPRIMS_MG: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "CUMLPRIMS_MG: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "CUMLPRIMS_MG: Enable nvtx markers: ${NVTX}")
message(VERBOSE "CUMLPRIMS_MG: Statically link the CUDA math libraries: ${CUDA_STATIC_MATH_LIBRARIES}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "CUMLPRIMS_MG: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "CUMLPRIMS_MG: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

#############################################################################
# - compiler options ---------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_MATH_LIBRARIES)
  set(_ctk_static_suffix "_static")
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ON)

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cumlprims_mg-exports
    INSTALL_EXPORT_SET cumlprims_mg-exports
    )
include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# - Requirements ----------------------------------------------------------------------------------

if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(VERBOSE "CUMLPRIMS_MG: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif(OPENMP_FOUND)
endif(NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})

# add third party dependencies using CPM
rapids_cpm_init()
rapids_cmake_install_lib_dir(lib_dir)

include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_raft.cmake)

if(BUILD_TESTS)
  find_package(MPI REQUIRED)
  include(cmake/thirdparty/get_nccl.cmake)
  include(cmake/thirdparty/get_gtest.cmake)
endif()


##############################################################################
# - clang-format targets -----------------------------------------------------

# find_package(ClangFormat 8.0.0 EXACT REQUIRED)
# TODO: enable this when we are ready!
# https://github.com/rapidsai/cumlprims_mg/issues/14
#find_package(ClangTidy 8.0.0 EXACT REQUIRED)


###################################################################################################
# - clang-format targets --------------------------------------------------------------------------

# add_clang_format(
#   DSTDIR ${PROJECT_BINARY_DIR}/clang-format
#   SRCDIR ${PROJECT_SOURCE_DIR})


###################################################################################################
# - build libcumlprims shared library -------------------------------------------------------------

add_library(cumlprims_mg
  src_prims_opg/linalg/eig.cu
  src_prims_opg/linalg/gemm.cu
  src_prims_opg/linalg/qr.cu
  src_prims_opg/linalg/qr_based_svd.cu
  src_prims_opg/matrix/matrix_utils.cu
  src_prims_opg/matrix/part_descriptor.cu
  src_prims_opg/matrix/math.cu
  src_prims_opg/linalg/mean_squared_error.cu
  src_prims_opg/linalg/mm_aTa.cu
  src_prims_opg/linalg/mv_aTb.cu
  src_prims_opg/linalg/lstsq.cu
  src_prims_opg/linalg/svd.cu
  src_prims_opg/linalg/norm.cu
  src_prims_opg/stats/mean.cu
  src_prims_opg/stats/mean_center.cu
  src_prims_opg/stats/stddev.cu
  src_prims_opg/stats/cov.cu
  src_prims_opg/stats/mean_center.cu
  src_prims_opg/stats/mean.cu
)
add_library(cumlprims_mg::cumlprims_mg ALIAS cumlprims_mg)

set_target_properties(cumlprims_mg
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cumlprims_mg
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUMLPRIMS_MG_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUMLPRIMS_MG_CUDA_FLAGS}>"
)

target_compile_definitions(cumlprims_mg
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUMLPRIMS_MG_CXX_DEFINITIONS}>"
                   "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUMLPRIMS_MG_CUDA_DEFINITIONS}>>"
)

set(CUMLPRIMS_MG_PUBLIC_INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${CUMLPRIMS_MG_SOURCE_DIR}/include>"
    )

target_include_directories(cumlprims_mg
           PUBLIC      ${CUMLPRIMS_MG_PUBLIC_INCLUDE_DIRECTORIES}
           PRIVATE     "$<BUILD_INTERFACE:${CUMLPRIMS_MG_SOURCE_DIR}/src_prims_opg>"
           INTERFACE   "$<INSTALL_INTERFACE:include>")


target_link_libraries(cumlprims_mg
    PUBLIC
      rmm::rmm
      raft::raft
    PRIVATE
      CUDA::cublas${_ctk_static_suffix}
      CUDA::curand${_ctk_static_suffix}
      CUDA::cusolver${_ctk_static_suffix}
      CUDA::cusparse${_ctk_static_suffix}
  )

if(OPENMP_FOUND)
  target_link_libraries(cumlprims_mg
    PRIVATE
      OpenMP::OpenMP_CXX
      pthread
    )
endif()

if(NVTX)
  target_link_libraries(cumlprims_mg PRIVATE CUDA::nvtx3)
endif()

###################################################################################################
# - build opg test executable ----------------------------------------------------------------

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

###################################################################################################
# - install targets -------------------------------------------------------------------------------

include(CPack)

install(TARGETS cumlprims_mg
        DESTINATION ${lib_dir}
        EXPORT cumlprims_mg-exports
        )

include(GNUInstallDirs)
install(DIRECTORY include/cumlprims
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

set(doc_string [=[
]=])

set(code_string [=[
]=])

rapids_export(
  INSTALL cumlprims_mg
  EXPORT_SET cumlprims_mg-exports
  GLOBAL_TARGETS cumlprims_mg::cumlprims_mg
  NAMESPACE cumlprims_mg::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD cumlprims_mg
  EXPORT_SET cumlprims_mg-exports
  GLOBAL_TARGETS cumlprims_mg::cumlprims_mg
  NAMESPACE cumlprims_mg::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

##################################################################################################
# - doxygen targets -------------------------------------------------------------------------------

# include(cmake/doxygen.cmake)
# add_doxygen_target(IN_DOXYFILE src_prims/Doxyfile.in
#   OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#   CWD ${CMAKE_CURRENT_BINARY_DIR})
