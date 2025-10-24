#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cuml build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean libcumlprims tests -v -g -n --allgpuarch -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean         - remove all existing build artifacts and configuration (start over).
   libcumlprims  - build the libcumlprims C++ code.
   tests         - build the C++ (OPG) tests.
 and <flag> is:
   -v            - verbose build mode
   -g            - build for debug
   -n            - no install step
   --allgpuarch  - build for all supported GPU architectures
   -h            - print this text

"
LIBCUMLPRIMS_BUILD_DIR=${REPODIR}/cpp/build
BUILD_DIRS="${LIBCUMLPRIMS_BUILD_DIR} ${CUML_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_ALL_GPU_ARCH=0
CLEAN=0
BUILD_TESTS=OFF

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg tests; then
    BUILD_TESTS=ON
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d ${bd} ]; then
        find ${bd} -mindepth 1 -delete
        rmdir ${bd} || true
    fi
    done
fi


################################################################################
# Configure for building all C++ targets
if (( ${NUMARGS} == 0 )) || hasArg libcumlprims; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUMLPRIMS_MG_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUMLPRIMS_MG_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    mkdir -p ${LIBCUMLPRIMS_BUILD_DIR}
    cd ${LIBCUMLPRIMS_BUILD_DIR}

    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUMLPRIMS_MG_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
          -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE \
          -DBUILD_TESTS=${BUILD_TESTS} \
          ..

fi

cd ${LIBCUMLPRIMS_BUILD_DIR}
TARGET_FLAG=""
if [ -n "${INSTALL_TARGET}" ]; then
  TARGET_FLAG="--target ${INSTALL_TARGET}"
fi
cmake --build ${LIBCUMLPRIMS_BUILD_DIR} -j${PARALLEL_LEVEL} ${build_args} ${TARGET_FLAG} ${VERBOSE_FLAG}
