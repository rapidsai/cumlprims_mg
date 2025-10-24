#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_gtest VERSION)

    if(TARGET GTest::gtest)
        return()
    endif()

    rapids_cpm_find(GTest ${VERSION}
        GLOBAL_TARGETS  gest gtest_main GTest::gtest GTest::gtest_main
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/google/googletest.git
            GIT_TAG         release-${VERSION}
            GIT_SHALLOW     TRUE
            OPTIONS         "INSTALL_GTEST OFF"
            # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
            FIND_PACKAGE_ARGUMENTS "CONFIG"
    )

    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endfunction()

set(CUMLPRIMS_MG_MIN_VERSION_gtest 1.10.0)

find_and_configure_gtest(${CUMLPRIMS_MG_MIN_VERSION_gtest})
