#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUMLPRIMS_MG_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas)
endif()

list(APPEND CUMLPRIMS_MG_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
# list(APPEND CUMLPRIMS_MG_CUDA_FLAGS -Werror=cross-execution-space-call)
# list(APPEND CUMLPRIMS_MG_CUDA_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

if(DISABLE_DEPRECATION_WARNINGS)
    list(APPEND CUMLPRIMS_MG_CXX_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
    list(APPEND CUMLPRIMS_MG_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
endif()

# make sure we produce smallest binary size
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
rapids_cuda_enable_fatbin_compression(VARIABLE CUMLPRIMS_MG_CUDA_FLAGS TUNE_FOR rapids)

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
if(CUDA_ENABLE_LINEINFO)
    list(APPEND CUMLPRIMS_MG_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(VERBOSE "CUMLPRIMS_MG: Building with debugging flags")
    list(APPEND CUMLPRIMS_MG_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()
