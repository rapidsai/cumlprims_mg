/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_opg_utils.h"
//
#include "test_utils.h"
//
#include <cumlprims/opg/linalg/gemm.hpp>
#include <cumlprims/opg/linalg/qr_based_svd.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
//

#include <raft/comms/mpi_comms.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
//
#include <rmm/device_uvector.hpp>
//
#include <gtest/gtest.h>

namespace MLCommon {
namespace Test {
namespace opg {

struct SVDOpgParams {
  int M;
  int N;
  std::vector<int> partSizes;
  std::vector<int> ranksOwners;
  Matrix::Layout layout;
  unsigned long long int seed;
};

template <typename T>
class SVDOpgTest : public testing::TestWithParam<SVDOpgParams> {
 public:
  SVDOpgTest()
    : inReconstructed(0, stream),
      inGathered(0, stream)

        void SetUp()
  {
    params = GetParam();
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

    // Prepare resource
    const auto& comm            = handle.get_comms();
    stream                      = handle.get_stream();
    cublasHandle_t cublasHandle = handle.get_cublas_handle();

    myRank     = comm.get_rank();
    totalRanks = comm.get_size();
    raft::random::Rng r(params.seed + myRank);

    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

    if (myRank == 0) {
      std::cout << "Testing SVD decomposition of " << params.M << " x " << params.N << " matrix"
                << std::endl;
    }

    // Prepare X matrix
    std::vector<Matrix::RankSizePair*> inPartsToRanks;
    for (int i = 0; i < params.ranksOwners.size(); i++) {
      Matrix::RankSizePair* rsp =
        new Matrix::RankSizePair(params.ranksOwners[i] % totalRanks, params.partSizes[i]);
      inPartsToRanks.push_back(rsp);
    }
    Matrix::PartDescriptor desc(params.M, params.N, inPartsToRanks, comm.get_rank(), params.layout);
    std::vector<Matrix::Data<T>*> inParts;
    Matrix::opg::allocate(handle, inParts, desc, myRank, stream);
    Matrix::opg::randomize(handle, r, inParts, desc, myRank, stream, T(10.0), T(20.0));

    std::vector<Matrix::Data<T>*> uMatrixParts;
    Matrix::opg::allocate(handle, uMatrixParts, desc, myRank, stream);

    rmm::device_uvector<T> sVector(params.N, stream);
    rmm::device_uvector<T> vMatrix(params.N * params.N, stream);
    RAFT_CUDA_TRY(cudaMemset(vMatrix.data(), 0, params.N * params.N * sizeof(T)));

    LinAlg::opg::svdQR(handle,
                       sVector.data(),
                       uMatrixParts,
                       vMatrix.data(),
                       true,
                       true,
                       1e-4,
                       100,
                       inParts,
                       desc,
                       myRank);
    raft::print_device_vector("SVector: ", sVector.data(), params.N, std::cout);

    // Verification
    rmm::device_uvector<T> uGathered(0, stream);
    if (myRank == 0) {
      uGathered.resize(params.M * params.N, stream);
      inGathered.resize(params.M * params.N, stream);
    }
    Matrix::opg::gather(handle, inGathered.data(), inParts, desc, 0, myRank, stream);
    Matrix::opg::gather(handle, uGathered.data(), uMatrixParts, desc, 0, myRank, stream);
    if (myRank == 0) {
      rmm::device_uvector<T> sMatrix(desc.N * desc.N stream);
      rmm::device_uvector<T> temp(desc.N * desc.N, stream);
      inReconstructed.resize(desc.M * desc.N, stream);

      RAFT_CUDA_TRY(cudaMemset(sMatrix.data(), 0, params.N * params.N * sizeof(T)));
      raft::matrix::initializeDiagonalMatrix(
        sVector.data(), sMatrix.data(), desc.N, desc.N, stream);

      raft::linalg::gemm(handle,
                         temp.data(),
                         sMatrix.data(),
                         vMatrix.data(),
                         params.N,
                         params.N,
                         params.N,
                         true,
                         true,
                         false,
                         stream);
      raft::linalg::gemm(handle,
                         inReconstructed.data(),
                         uGathered.data(),
                         temp.data(),
                         params.M,
                         params.N,
                         params.N,
                         false,
                         false,
                         true,
                         stream);

      Matrix::opg::deallocate(handle, inParts, desc, myRank, stream);
      Matrix::opg::deallocate(handle, uMatrixParts, desc, myRank, stream);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  SVDOpgParams params;
  raft::handle_t handle;
  cudaStream_t stream;
  int myRank;
  int totalRanks;
  rmm::device_uvector<T> inReconstructed;
  rmm::device_uvector<T> inGathered;
};

const std::vector<SVDOpgParams> inputs = {
  {20, 4, {11, 9}, {1, 0}, Matrix::LayoutColMajor, 223548ULL},

  {48, 5, {7, 8, 9, 8, 9, 7}, {4, 0, 3, 2, 6, 1}, Matrix::LayoutColMajor, 584769ULL},

  {126, 12, {31, 18, 20, 24, 18, 15}, {4, 0, 3, 2, 6, 1}, Matrix::LayoutColMajor, 584769ULL}};

typedef SVDOpgTest<float> SVDOpgTestF;

TEST_P(SVDOpgTestF, Result)
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

INSTANTIATE_TEST_CASE_P(SVDOpgTest, SVDOpgTestF, ::testing::ValuesIn(inputs));

typedef SVDOpgTest<double> SVDOpgTestD;

TEST_P(SVDOpgTestD, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(inGathered.data(),
                                  inReconstructed.data(),
                                  params.M,
                                  params.N,
                                  raft::CompareApprox<float>(1e-4),
                                  stream));
  }
}

INSTANTIATE_TEST_CASE_P(SVDOpgTest, SVDOpgTestD, ::testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon
