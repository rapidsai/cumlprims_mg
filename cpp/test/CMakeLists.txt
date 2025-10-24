#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================


# (keep the filenames in alphabetical order)
add_executable(mg_test
    opg/eig.cu
    opg/gemm.cu
    opg/main.cu
    opg/qr.cu
    opg/qr_based_svd.cu
)

set_target_properties(mg_test
PROPERTIES BUILD_RPATH                         "\$ORIGIN"
           # set target compile options
           CXX_STANDARD                        14
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       14
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(mg_test
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUMLPRIMS_MG_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUMLPRIMS_MG_CUDA_FLAGS}>"
)

target_include_directories(mg_test
    PUBLIC
        "$<BUILD_INTERFACE:${MPI_CXX_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/test>"
        "$<BUILD_INTERFACE:${CUMLPRIMS_MG_SOURCE_DIR}/src_prims_opg>"
        "$<BUILD_INTERFACE:${CUMLPRIMS_MG_SOURCE_DIR}/test/opg>"
)

target_link_libraries(mg_test
 PUBLIC
    GTest::gtest
    GTest::gtest_main
    NCCL::NCCL
    ${MPI_CXX_LIBRARIES}
    cumlprims_mg::cumlprims_mg
  )
