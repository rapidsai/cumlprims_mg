/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
#include "test_opg_utils.h"
//
#include "test_utils.h"
//

#include <cumlprims/opg/linalg/eig.hpp>
#include <cumlprims/opg/linalg/gemm.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
//
#include <raft/comms/mpi_comms.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
//
#include <rmm/device_uvector.hpp>
//
#include <gtest/gtest.h>

namespace MLCommon {
namespace Test {
namespace opg {

struct EigOpgParams {
  int N;
  std::vector<int> partSizes;
  std::vector<int> ranksOwners;
  Matrix::Layout layout;
  unsigned long long int seed;
};

/**
 * @brief This kernel is used to initialize block of a matrix. Matrix layout is
 * assumed to be column major. We initialize it to
 *          row + col,
 * this expression evaluates to same value for (row, col) and (col, row) thus
 * generated matrix is symmetric.
 *
 * \param[out]  block        A buffer in device memory of size (N * N)
 *                           This buffer holds the output matrix.
 * \param[in]   N            Dimension of matrix (N x N).
 * \param[in]   startIndex   Startig row index of the part.
 * \param[in]   partSize     Number of rows in the part.
 */
template <typename T>
static __global__ void initializePart(T* part, size_t startIndex, int partSize, int N)
{
  int rowOffset = threadIdx.x + blockIdx.x * blockDim.x;
  for (; rowOffset < partSize; rowOffset += blockDim.x * gridDim.x) {
    int row = startIndex + rowOffset;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    for (; col < N; col += blockDim.y * gridDim.y) {
      part[col * partSize + rowOffset] = row + col;
    }
  }
}

template <typename T>
class EigOpgTest : public ::testing::TestWithParam<EigOpgParams> {
 public:
  EigOpgTest() : lhs(0, stream), rhs(0, stream) {}

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
      std::cout << "Testing Eigen decomposition of " << params.N << " x " << params.N << " matrix"
                << std::endl;
    }

    // Prepare in matrix
    std::vector<Matrix::RankSizePair*> inPartsToRanks;
    for (int i = 0; i < params.ranksOwners.size(); i++) {
      Matrix::RankSizePair* rsp =
        new Matrix::RankSizePair(params.ranksOwners[i] % totalRanks, params.partSizes[i]);
      inPartsToRanks.push_back(rsp);
    }
    Matrix::PartDescriptor desc(params.N, params.N, inPartsToRanks, comm.get_rank(), params.layout);
    std::vector<Matrix::Data<T>*> inParts;
    Matrix::opg::allocate(handle, inParts, desc, myRank, stream);

    std::vector<size_t> start_indices = desc.startIndices();
    for (int i = 0, localIndex = 0; i < desc.partsToRanks.size(); i++) {
      if (desc.partsToRanks[i]->rank == myRank) {
        dim3 threads(32, 32, 1);
        dim3 blocks(16, 4, 1);
        initializePart<T><<<blocks, threads>>>(
          inParts[localIndex]->ptr, start_indices[i], desc.partsToRanks[i]->size, params.N);
        localIndex++;
      }
    }

    rmm::device_uvector<T> eigenValues(params.N, stream);
    rmm::device_uvector<T> eigenVectors(params.N * params.N, stream);

    LinAlg::opg::eigDC(handle, eigenValues, eigenVectors, inParts, desc, myRank, stream);

    // Verification part
    // Verify if in * eigenVectors == eigenMatrix * eigenVectors
    // Here, eigenMatrix is a diagonal matrix with Eigen values as diagonal
    // We compute left hand side (LHS) and right hand side (RHS) of equation
    // separately.

    rmm::device_uvector<T> inGathered(0, stream);
    if (myRank == 0) { inGathered.resize(params.N * params.N, stream); }
    Matrix::opg::gather(handle, inGathered, inParts, desc, 0, myRank, stream);
    if (myRank == 0) {
      rhs.resize(params.N * params.N, stream);
      lhs.resize(params.N * params.N, stream);
      rmm::device_uvector<T> eigenMatrix(params.N * params.N, stream);
      RAFT_CUDA_TRY(
        cudaMemsetAsync(eigenMatrix.data(), 0, params.N * params.N * sizeof(T), stream));
      RAFT_CUDA_TRY(cudaMemsetAsync(lhs, 0, params.N * params.N * sizeof(T), stream));
      raft::matrix::initializeDiagonalMatrix(
        eigenValues, eigenMatrix.data(), params.N, params.N, stream);
      // Compute LHS = in * eigenVector
      raft::linalg::gemm(handle,
                         lhs,
                         inGathered,
                         eigenVectors,
                         params.N,
                         params.N,
                         params.N,
                         true,
                         false,
                         true,
                         stream);
      // Compute RHS = eigVectors * diagonal(eigMatrix)
      raft::linalg::gemm(handle,
                         rhs.data(),
                         eigenVectors,
                         eigenMatrix.data(),
                         params.N,
                         params.N,
                         params.N,
                         true,
                         true,
                         true,
                         stream);
    }
    Matrix::opg::deallocate(handle, inParts, desc, myRank, stream);
  }

 protected:
  EigOpgParams params;
  raft::handle_t handle;
  cudaStream_t stream;
  int myRank;
  int totalRanks;
  rmm::device_uvector<T> rhs;
  rmm::device_uvector<T> lhs;
};

const std::vector<EigOpgParams> inputs = {
  {5, {3, 2}, {1, 0}, Matrix::LayoutColMajor, 223548ULL},

  {48, {7, 8, 9, 8, 9, 7}, {4, 0, 3, 2, 6, 1}, Matrix::LayoutColMajor, 584769ULL},
};

typedef EigOpgTest<float> EigOpgTestF;

TEST_P(EigOpgTestF, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(
      lhs.data(), rhs.data(), params.N, params.N, raft::CompareApprox<float>(1e-3), stream));
  }
}

INSTANTIATE_TEST_CASE_P(EigOpgTest, EigOpgTestF, ::testing::ValuesIn(inputs));

typedef EigOpgTest<double> EigOpgTestD;

TEST_P(EigOpgTestD, Result)
{
  if (myRank == 0) {
    ASSERT_TRUE(raft::devArrMatch(
      lhs.data(), rhs.data(), params.N, params.N, raft::CompareApprox<float>(1e-3), stream));
  }
}

INSTANTIATE_TEST_CASE_P(EigOpgTest, EigOpgTestD, ::testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon
