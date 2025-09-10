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
//
#include "test_opg_utils.h"
//
#include "test_utils.h"
//
#include <cumlprims/opg/linalg/gemm.hpp>
#include <cumlprims/opg/linalg/qr.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
//
#include <raft/comms/mpi_comms.hpp>
#include <raft/linalg/gemm.cuh>
//
#include <rmm/device_uvector.hpp>
//
#include <gtest/gtest.h>

namespace MLCommon {
namespace Test {
namespace opg {

struct QROpgParams {
  int M;
  int N;
  std::vector<int> partSizes;
  std::vector<int> ranksOwners;
  Matrix::Layout layout;
  unsigned long long int seed;
};

template <typename T>
class QROpgTest : public testing::TestWithParam<QROpgParams> {
 public:
  QROpgTest() : inGathered(0, stream), inReconstructed(0, stream) {}

  void SetUp()
  {
    params = GetParam();
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

    // Prepare resource
    const auto& comm = handle.get_comms();
    stream           = handle.get_stream();

    myRank     = comm.get_rank();
    totalRanks = comm.get_size();
    raft::random::Rng r(params.seed + myRank);

    if (myRank == 0) {
      std::cout << "Testing QR factorization of " << params.M << " layout " << params.N << " matrix"
                << std::endl;
    }

    // Prepare in matrix
    std::vector<Matrix::RankSizePair*> inPartsToRanks;
    for (int i = 0; i < params.ranksOwners.size(); i++) {
      Matrix::RankSizePair* rsp =
        new Matrix::RankSizePair(params.ranksOwners[i] % totalRanks, params.partSizes[i]);
      inPartsToRanks.push_back(rsp);
    }
    Matrix::PartDescriptor desc(params.M, params.N, inPartsToRanks, comm.get_rank(), params.layout);
    std::vector<Matrix::Data<T>*> inParts;
    Matrix::opg::allocate(handle, inParts, desc, myRank, stream);
    Matrix::opg::randomize(handle, r, inParts, desc, myRank, stream, T(-10.0), T(10.0));

    std::vector<Matrix::Data<T>*> outQParts;
    Matrix::opg::allocate(handle, outQParts, desc, myRank, stream);

    // Full R matrix duplicated across ranks
    rmm::device_uvector<T> outR(params.N * params.N, stream);
    RAFT_CUDA_TRY(cudaMemset(outR.data(), 0, params.N * params.N * sizeof(T)));
    LinAlg::opg::qrDecomp(handle, outQParts, outR.data(), inParts, desc, myRank);

    rmm::device_uvector<T> qGathered(0, stream);
    if (myRank == 0) {
      qGathered.resize(params.M * params.N, stream);
      inGathered.resize(params.M * params.N, stream);
      inReconstructed.resize(params.M * params.N, stream);
    }

    Matrix::opg::gather(handle, qGathered.data(), outQParts, desc, 0, myRank, stream);
    Matrix::opg::gather(handle, inGathered.data(), inParts, desc, 0, myRank, stream);
    if (myRank == 0) {
      raft::linalg::gemm(handle,
                         inReconstructed.data(),
                         qGathered.data(),
                         outR.data(),
                         params.M,
                         params.N,
                         params.N,
                         false,
                         false,
                         true,
                         stream);
    }
    // Clean-up
    Matrix::opg::deallocate(handle, inParts, desc, myRank, stream);
    Matrix::opg::deallocate(handle, outQParts, desc, myRank, stream);
    for (int i = 0; i < desc.partsToRanks.size(); i++) {
      delete desc.partsToRanks[i];
    }
  }

 protected:
  QROpgParams params;
  raft::handle_t handle;
  cudaStream_t stream;
  int myRank;
  int totalRanks;
  rmm::device_uvector<T> inGathered;
  rmm::device_uvector<T> inReconstructed;
};

const std::vector<QROpgParams> inputs = {
  {20, 4, {11, 9}, {1, 0}, Matrix::LayoutColMajor, 223548ULL},

  {48, 5, {7, 8, 9, 8, 9, 7}, {4, 0, 3, 2, 6, 1}, Matrix::LayoutColMajor, 584769ULL},

  {126, 12, {31, 18, 20, 24, 18, 15}, {4, 0, 3, 2, 6, 1}, Matrix::LayoutColMajor, 584769ULL},

  {2141, 43, {345, 404, 421, 432, 539}, {3, 7, 12, 6, 0}, Matrix::LayoutColMajor, 733704ULL},

  {3940,
   9,
   {270, 380, 327, 294, 418, 272, 244, 334, 351, 263, 273, 514},
   {12, 10, 7, 9, 5, 4, 5, 1, 2, 1, 8, 12},
   Matrix::LayoutColMajor,
   259852ULL},

  {4100,
   6,
   {374, 385, 392, 349, 469, 277, 374, 430, 383, 452, 215},
   {2, 0, 14, 2, 10, 6, 4, 8, 10, 11, 3},
   Matrix::LayoutColMajor,
   578726ULL},
};

typedef QROpgTest<float> QROpgTestF;

TEST_P(QROpgTestF, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(inGathered.data(),
                                  inReconstructed.data(),
                                  params.M,
                                  params.N,
                                  raft::CompareApprox<float>(1e-3),
                                  stream));
  }
}

INSTANTIATE_TEST_CASE_P(QROpgTest, QROpgTestF, testing::ValuesIn(inputs));

typedef QROpgTest<double> QROpgTestD;

TEST_P(QROpgTestD, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(inGathered.data(),
                                  inReconstructed.data(),
                                  params.M,
                                  params.N,
                                  raft::CompareApprox<double>(1e-4),
                                  stream));
  }
}

INSTANTIATE_TEST_CASE_P(QROpgTest, QROpgTestD, testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon
