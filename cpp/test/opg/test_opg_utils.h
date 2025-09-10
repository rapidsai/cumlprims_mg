/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "comm_utils.h"
//
#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
//
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
//
#include <mpi.h>
//
#include <gtest/gtest.h>

namespace MLCommon {
namespace Test {
namespace opg {
/**
 * @brief A naive attempt at creating different inputs in each of the ranks
 * @param seed the input seed (typically as defined in the test params)
 * @return a unique rank across all ranks
 */
inline unsigned long long rankBasedSeed(const raft::comms::comms_t& comm, unsigned long long seed)
{
  int myRank = comm.get_rank();
  return seed + myRank;
}

/** checks whether the current process is the root or not */
inline bool amIroot(const raft::comms::comms_t& comm, int rootRank = 0)
{
  int myRank = comm.get_rank();
  return myRank == rootRank;
}

/**
 *
 * @brief Testing environment to handle googletest runs
 * @note Inspired from:
 * http://www.parresianz.com/mpi/c++/mpi-unit-testing-googletests-cmake/
 */
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp()
  {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nGpus;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&nGpus));

    ASSERT(
      nGpus >= size, "Number of GPUs are lesser than MPI ranks! ngpus=%d, nranks=%d", nGpus, size);

    RAFT_CUDA_TRY(cudaSetDevice(rank));
  }

  void TearDown() { MPI_Finalize(); }
};

};  // end namespace opg
};  // end namespace Test
};  // end namespace MLCommon
