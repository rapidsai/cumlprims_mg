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
// cumlprims header
#include "test_opg_utils.h"
// cuml header
#include "test_utils.h"
//
#include <cumlprims/opg/linalg/gemm.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
//

#include <raft/comms/mpi_comms.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/util/cudart_utils.hpp>
//
#include <rmm/device_uvector.hpp>
//
#include <gtest/gtest.h>

namespace MLCommon {
namespace Test {
namespace opg {

struct GemmOpgParams {
  int M;
  int N;
  int K;
  std::vector<int> xPartSizes;
  std::vector<int> xRanksOwners;
  std::vector<int> yPartSizes;
  std::vector<int> yRanksOwners;
  Matrix::Layout xLayout;
  Matrix::Layout yLayout;
  Matrix::Layout zLayout;
  unsigned long long int seed;
};

template <typename T>
class GemmOpgTest : public ::testing::TestWithParam<GemmOpgParams> {
 public:
  GemmOpgTest() : zGatheredRef(0, stream), zGathered(0, stream) {}

  void SetUp()
  {
    params = GetParam();

    handle = new raft::handle_t();

    RAFT_CUDA_TRY(cudaGetLastError());

    raft::comms::initialize_mpi_comms(handle, MPI_COMM_WORLD);
    const auto& comm    = handle->get_comms();
    cudaStream_t stream = handle->get_stream();
    myRank              = comm.get_rank();
    totalRanks          = comm.get_size();
    raft::random::Rng r(params.seed + myRank);

    comm.barrier();

    // Prepare X matrix
    std::vector<Matrix::RankSizePair*> inXPartsToRanks;
    for (int i = 0; i < params.xRanksOwners.size(); i++) {
      Matrix::RankSizePair* rsp =
        new Matrix::RankSizePair(params.xRanksOwners[i] % totalRanks, params.xPartSizes[i]);
      inXPartsToRanks.push_back(rsp);
    }
    Matrix::PartDescriptor inXDesc(params.M, params.K, inXPartsToRanks, myRank, params.xLayout);
    std::vector<Matrix::Data<T>*> inXParts;
    Matrix::opg::allocate(*handle, inXParts, inXDesc, myRank, stream);
    Matrix::opg::randomize(*handle, r, inXParts, inXDesc, myRank, stream, 1.0, 2.0);

    // Prepare Y matrix
    std::vector<Matrix::RankSizePair*> inYPartsToRanks;
    for (int i = 0; i < params.yRanksOwners.size(); i++) {
      Matrix::RankSizePair* rsp =
        new Matrix::RankSizePair(params.yRanksOwners[i] % totalRanks, params.yPartSizes[i]);
      inYPartsToRanks.push_back(rsp);
    }
    Matrix::PartDescriptor inYDesc(params.K, params.N, inYPartsToRanks, myRank, params.yLayout);
    std::vector<Matrix::Data<T>*> inYParts;
    Matrix::opg::allocate(*handle, inYParts, inYDesc, myRank, stream);
    Matrix::opg::randomize(*handle, r, inYParts, inYDesc, myRank, stream, 1.0, 2.0);

    // Prepapre Z matrix
    std::vector<Matrix::RankSizePair*> outZPartsToRanks;
    for (int i = 0; i < params.xRanksOwners.size(); i++) {
      Matrix::RankSizePair* rsp =
        new Matrix::RankSizePair(params.xRanksOwners[i] % totalRanks, params.xPartSizes[i]);
      outZPartsToRanks.push_back(rsp);
    }
    Matrix::PartDescriptor outZDesc(params.M, params.N, outZPartsToRanks, myRank, params.zLayout);
    std::vector<Matrix::Data<T>*> outZParts;
    Matrix::opg::allocate(*handle, outZParts, outZDesc, myRank, stream);

    LinAlg::opg::gemm(
      *handle, outZParts, outZDesc, inXParts, inXDesc, inYParts, inYDesc, myRank, stream);

    comm.sync_stream(stream);

    // Verification part
    // Gather X, Y and Z at rank 0
    rmm::device_uvector<T> xGathered(0, stream);
    rmm::device_uvector<T> yGathered(0, stream);
    if (myRank == 0) {
      xGathered.resize(params.M * params.K, stream);
      yGathered.resize(params.K * params.N, stream);
      zGatheredRef.resize(params.M * params.N, stream);
      zGathered.resize(params.M * params.N, stream);
    }

    Matrix::opg::gather(*handle, xGathered, inXParts, inXDesc, 0, myRank, stream);
    Matrix::opg::gather(*handle, yGathered, inYParts, inYDesc, 0, myRank, stream);
    Matrix::opg::gather(*handle, zGathered, outZParts, outZDesc, 0, myRank, stream);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    if (myRank == 0) {
      raft::linalg::gemm(*handle,
                         zGatheredRef,
                         xGathered,
                         yGathered,
                         params.M,
                         params.N,
                         params.K,
                         false,
                         false,
                         false,
                         stream);
    }

    // Clean up
    Matrix::opg::deallocate(*handle, inXParts, inXDesc, myRank, stream);
    Matrix::opg::deallocate(*handle, inYParts, inYDesc, myRank, stream);
    Matrix::opg::deallocate(*handle, outZParts, outZDesc, myRank, stream);

    for (int i = 0; i < inXDesc.partsToRanks.size(); i++) {
      delete inXDesc.partsToRanks[i];
    }
    for (int i = 0; i < inYDesc.partsToRanks.size(); i++) {
      delete inYDesc.partsToRanks[i];
    }
    for (int i = 0; i < outZDesc.partsToRanks.size(); i++) {
      delete outZDesc.partsToRanks[i];
    }

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  void TearDown()
  {
    delete handle;
    RAFT_CUDA_TRY(cudaGetLastError());
  }

 protected:
  GemmOpgParams params;
  raft::handle_t* handle;
  rmm::device_uvector<T> zGatheredRef;
  rmm::device_uvector<T> zGathered;
  int myRank;
  int totalRanks;
};

const std::vector<GemmOpgParams> inputs = {
  {12,
   16,
   8,
   {4, 4, 4},
   {0, 1, 0},
   {6, 2},
   {1, 0},
   Matrix::LayoutColMajor,
   Matrix::LayoutColMajor,
   Matrix::LayoutColMajor,
   1234ULL},

  {64,
   32,
   25,
   {9, 2, 10, 14, 18, 11},
   {4, 0, 3, 2, 6, 1},
   {4, 3, 6, 6, 4, 2},
   {7, 4, 5, 3, 2, 1},
   Matrix::LayoutColMajor,
   Matrix::LayoutColMajor,
   Matrix::LayoutRowMajor,
   584769ULL},

  {789,
   415,
   87,
   {63, 60, 46, 73, 84, 73, 53, 54, 79, 63, 86, 55},
   {6, 6, 15, 0, 15, 9, 5, 2, 1, 8, 14, 15},
   {14, 21, 12, 19, 21},
   {0, 6, 5, 8, 13},
   Matrix::LayoutColMajor,
   Matrix::LayoutRowMajor,
   Matrix::LayoutColMajor,
   252439ULL},

  {682,
   708,
   200,
   {40, 47, 63, 44, 38, 40, 50, 50, 36, 44, 63, 49, 58, 60},
   {10, 12, 12, 14, 8, 3, 4, 5, 5, 2, 4, 9, 1, 5},
   {14, 15, 19, 20, 15, 14, 13, 15, 13, 22, 11, 29},
   {6, 8, 13, 5, 5, 8, 7, 11, 8, 7, 1, 3},
   Matrix::LayoutColMajor,
   Matrix::LayoutColMajor,
   Matrix::LayoutRowMajor,
   561550ULL}};

typedef GemmOpgTest<float> GemmOpgTestF;

TEST_P(GemmOpgTestF, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(
      zGatheredRef, zGathered, params.M, params.N, raft::CompareApprox<float>(1e-4)));
  }
}

INSTANTIATE_TEST_CASE_P(GemmOpgTest, GemmOpgTestF, ::testing::ValuesIn(inputs));

typedef GemmOpgTest<double> GemmOpgTestD;

TEST_P(GemmOpgTestD, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(
      zGatheredRef, zGathered, params.M, params.N, raft::CompareApprox<float>(1e-6)));
  }
}

INSTANTIATE_TEST_CASE_P(GemmOpgTest, GemmOpgTestD, ::testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon
