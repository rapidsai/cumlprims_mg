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
#include <cumlprims/opg/linalg/gemm.hpp>
#include <cumlprims/opg/linalg/qr.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
#include <raft/linalg/svd.cuh>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * \brief       Multi-GPU SVD decomposition for tall (rows >= columns) matrices
 * \tparam      T             Data-type for in, U, S and V matrices.
 * \param       h             cuML handle object.
 * \param[out]  sVector       A vector of size 1 x N with eigen values of
                              in matrix.
 * \param[out]  uParts        Parts of output U matrix from SVD decomposition
                              with size M x N. It is distributed among ranks.
                              Descriptor desc describes the matrix.
 * \param[out]  vMatrix       Full output V matrix from SVD decomposition with
                              size N x N. It is duplicated on all ranks.
 * \param[in]   genUMatrix    Currently ignored.
                              U matrix is generated only if this is true.
 * \param[in]   genVMatrix    Currently ignored.
                              V matrix is generated only if this is true.
 * \param[in]   tolerance     Error tolerance used for single GPU SVD.
                              Algorithm stops when the error is below
                              tolerance
 * \param[in]   maxSweeps     Number of sweeps in the single GPU SVD using
                              Jacobi algorithm. More sweeps provide better
                              accuracy.
 * \parms[in]   inParts       Parts of the tall input matrix, distributed among
                              ranks. The size of in matrix is M x N,
                              where M >> N.
 * \param[in]   desc          Discriptor of in matrix (inParts) and U matrix
                              (uParts).
 * \param       myRank        MPI rank of the process
 */
template <typename T>
void svdQR(const raft::handle_t& h,
           T* sVector,
           std::vector<Matrix::Data<T>*>& uParts,
           T* vMatrix,
           bool genUMatrix,
           bool genVMatrix,
           T tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<T>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank)
{
  /*
  Input matrix: 'inParts' with size M x N. The 'in' matrix is row-wise
  sharded in to 'numParts' parts, as shown below. Each part can have any number
  of number of rows and it is given by desc.partsToRanks[i]->size. However
  each part has fixed number of columns N. For this SVD algorithm to work, all
  the parts must be tall (rows > columns), in turn making M >> N.

  | <----------------- N columns ---------------------> |

  .-----------------------------------------------------.   ------
  |                       Part 0                        |     ^
  |            partsToRanks[0]->size x N                |     |
  +-----------------------------------------------------+     |
  |                       Part 1                        |     |
  |            partsToRanks[1]->size x N                |     |
  +-----------------------------------------------------+     |
  |                                                     |     |

  *                          *                          *
  *                          *                          *   M rows
  *                          *                          *

  |                                                     |     |
  +-----------------------------------------------------+     |
  |               Part (numParts - 1)                   |     |
  |      partsToRanks[numParts - 1]->size x N           |     v
  +-----------------------------------------------------+   ------

  SVD decomposition factors of inParts are computed in this  function.
      inParts = uMatrix * diag(sVec) * tr(vMatrix)
  Where, diag(sVec) is a diagonal matrix with sVec as diagonal values and
    tr(vMatrix) is transpose of vMatrix.
  Details:
    First multi-gpu QR method is used to factor inParts into qParts and
    rMatrix.
    inParts = qParts * rMatrix

    On rank 0
      SVD decomposition of rMatrix is performed
      rMatrix = uOfR * sOfR * vOfR'
      sOfR is same as sVec and vMatrix is same output vOfR.
    Broadcast vMatrix to other ranks from rank 0
    On each rank perform matrix multiplication to get uParts
      uParts = qParts * uOfR
  */
  const auto& comm = h.get_comms();

  cudaStream_t userStream = h.get_stream();

  size_t N     = desc.N;
  int numParts = desc.partsToRanks.size();

  size_t minPartSize = desc.M;  // desc.M is sum of number of rows in all parts
                                // therefore obviously bigger than all parts
  for (int i = 0; i < numParts; i++) {
    if (minPartSize > desc.partsToRanks[i]->size) { minPartSize = desc.partsToRanks[i]->size; }
  }

  ASSERT(desc.M >= desc.N,
         "MLCommon::LinAlg::opg::SVD: Number of rows of"
         " input matrix can not be less than number of columns");
  ASSERT(minPartSize >= desc.N,
         "MLCommon::LinAlg::opg::SVD: Number of rows of "
         " any input matrix block can not be less than number of columns in"
         " the block");
  ASSERT(desc.layout == Matrix::Layout::LayoutColMajor,
         "MLCommon::LinAlg::opg::SVD: Intra block layout other than column"
         " major is not supported.");

  std::vector<Matrix::Data<T>*> qParts;
  Matrix::opg::allocate(h, qParts, desc, myRank, userStream);
  rmm::device_uvector<T> rMatrix(N * N, userStream);
  rmm::device_uvector<T> uOfR(N * N, userStream);

  RAFT_CUDA_TRY(cudaMemsetAsync(rMatrix.data(), 0, N * N * sizeof(T), userStream));
  qrDecomp(h, qParts, rMatrix.data(), inParts, desc, myRank);

  if (myRank == 0) {
    raft::linalg::svdJacobi(h,
                            rMatrix.data(),
                            N,
                            N,
                            sVector,
                            uOfR.data(),
                            vMatrix,
                            genUMatrix,
                            genVMatrix,
                            tolerance,
                            maxSweeps,
                            userStream);
  }
  comm.bcast(sVector, N, 0, userStream);
  comm.bcast(uOfR.data(), N * N, 0, userStream);
  comm.bcast(vMatrix, N * N, 0, userStream);
  comm.sync_stream(userStream);

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      cudaStream_t stream = h.get_next_usable_stream();

      raft::linalg::gemm(h,
                         uParts[localId]->ptr,
                         qParts[localId]->ptr,
                         uOfR.data(),
                         desc.partsToRanks[i]->size,
                         N,
                         N,
                         true,
                         true,
                         true,
                         stream);
      localId++;
    }
  }
  h.sync_stream_pool();

  Matrix::opg::deallocate(h, qParts, desc, myRank, userStream);
}

// Instantiations
void svdQR(const raft::handle_t& h,
           float* sVector,
           std::vector<Matrix::Data<float>*>& uMatrixParts,
           float* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           float tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank)
{
  svdQR<float>(h,
               sVector,
               uMatrixParts,
               vMatrixParts,
               genUMatrix,
               genVMatrix,
               tolerance,
               maxSweeps,
               inParts,
               desc,
               myRank);
}

void svdQR(const raft::handle_t& h,
           double* sVector,
           std::vector<Matrix::Data<double>*>& uMatrixParts,
           double* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           double tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank)
{
  svdQR<double>(h,
                sVector,
                uMatrixParts,
                vMatrixParts,
                genUMatrix,
                genVMatrix,
                tolerance,
                maxSweeps,
                inParts,
                desc,
                myRank);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
