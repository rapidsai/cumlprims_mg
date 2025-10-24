/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cumlprims/opg/linalg/gemm.hpp>
#include <cumlprims/opg/linalg/qr.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/linalg/transpose.cuh>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * \brief This function repacks multiple blocks of matrix in to a single blocks.
 *  Input and output blocks are assumed to be in column major order.
 */
template <typename T>
static __global__ void r1Repack(T* inMatNR, T* inMatNRCorrected, int totalElements, int M, int totalBlocks)
{
  int inElemId = threadIdx.x + blockDim.x * blockIdx.x;

  if (inElemId < totalElements) {
    int partId                  = inElemId / (M * M);
    int blockOffset             = inElemId % (M * M);
    int inRow                   = blockOffset % M;
    int inCol                   = blockOffset / M;
    int outRow                  = partId * M + inRow;
    int outCol                  = inCol;
    int outElemId               = outCol * totalBlocks * M + outRow;
    inMatNRCorrected[outElemId] = inMatNR[inElemId];
  }
}

/**
 * \brief This function copies upper triangular part of in matrix to out matrix
 */
template <typename T>
static __global__ void copyUpperTriangle(T* out, T* in, int N)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int col = threadIdx.y + blockDim.y * blockIdx.y;
  if (row < N && col < N && row <= col) { out[col * N + row] = in[col * N + row]; }
}

/**
 * \brief    Multi-GPU QR decomposition for tall (rows > columns) matrices.
 *
 * This function implements multi-GPU QR decomposition for tall matrices
 * descibed in https://arxiv.org/abs/1301.1071.
 * \tparam      T               Data-type for in, Q and R matrices.
 * \param       h               cuML handle object.
 * \param[out]  outQParts       Parts of factor Q of input matrix QR
                                decomposition. Q matrix is distributed
                                among ranks according to desc description.
                                The size of Q matrix is same as size of 'in'
                                matrix, i.e. M x N.
 * \param[out]  outR            Factor R of input matrix QR decomposition. R is
                                N x N matrix and is duplicated on all ranks.
                                Note that, only upper triangular part of outR,
                                including diagonal, is modified. Lower
                                triangular part is left as it is.
 * \param[in]   inParts         The tall input matrix, distributed among ranks.
                                The size of 'in' matrix is M x N, where M > N.
 * \param[in]   desc            Description of input matrix. The output matrix
                                'outQParts' follows the same description.
 * \param       myRank          MPI rank of the process.
 */
template <typename T>
void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<T>*>& outQParts,
              T* outR,
              std::vector<Matrix::Data<T>*>& inParts,
              Matrix::PartDescriptor& desc,
              int myRank)
{
  /*
  Input matrix: 'in' with size M x N. The 'in' matrix is row-wise
  sharded in to 'numParts' parts, as shown below. Each part can have any number
  of number of rows and it is given by desc.partsToRanks[i]->size. However
  each part has fixed number of columns N. For this QR algorithm to work, all
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


  In first step, QR decomposition of each part is carried out in parallel
  across ranks.
  q1Parts[i], r1Parts[i] = QR(inParts[i])
  Here,
     inParts[i] size partsToRanks[i]->size x N
     q1Parts[i] size partsToRanks[i]->size x N
     r1Parts[i] size N x N
  All ranks send r1Parts to rank 0
  At rank 0
    r1Parts from all ranks is collected in r1Collected. r1Collected has
    numParts contagious parts, each part is N x N and layout column major.
    r1Collected is repacked to form matrix r1Corrected with single part of size
    (numParts * N) rows and N columns, with column major layout.
    QR decomposition of r1Corrected
    q2, r2 = qr(r1Corrected)
    Here,
      r1Corrected size numParts * N x N
      q2 size numParts * N x N in column major layout
      r2 size N x N
    Layout of q2 is changed to row major to enable row sharding
    q2RowMajor = transform(q2)
    q2RowMajor size numParts * N x N
    Parts of q2RowMajor needs to be scattered among all ranks. Part i of
    q2RowMajor goes to partsToRanks[i]->rank
  At each ranks
  Receive q2Parts[i] sent by rank 0
  Each q2Parts[i] is of size N x N. A rank can have more than one part and is
  given by numLocalParts.  size of q2Parts is numLocalParts * N * N.
  Each rank performs GEMM operation of q1Parts[i] and q2Parts[i] and the
  result is stored in return argument outQParts[i].
  temp = q1Parts[i] * q2Parts[i]
   q1Parts[i] size partsToRanks[i]->size x N
   q2Parts[i] size N x N
   temp size partsToRanks[i]->size x N
  outQParts[i] = finalQParts[i]
  Rank 0 broadcasts r2; Each rank receives it in return argument outR

  Various temporary buffer size

  At all ranks
    r1Parts         numLocalParts * N * N * sizeof(T)
    r2              N * N * sizeof(T)
    q2Parts         numLocalParts * N * N * sizeof(T)

  Only at Rank 0
    r1Collected     numParts * N * N *sizeof(T)
    r1Corrected     numParts * N * N *sizeof(T)
    q2              numParts * N * N *sizeof(T)
    q2RowMajor      numParts * N * N *sizeof(T)
*/

  const auto& comm = h.get_comms();

  cudaStream_t userStream = h.get_stream();

  size_t N          = desc.N;
  int numParts      = desc.partsToRanks.size();
  int numLocalParts = desc.totalBlocksOwnedBy(myRank);

  std::vector<size_t> partSizes;
  size_t maxPartSize = 0;
  size_t minPartSize = desc.M;  // desc.M is sum of number of rows in all parts
                                // therefore obviously bigger than all parts
  for (int i = 0; i < numParts; i++) {
    partSizes.push_back(desc.partsToRanks[i]->size);
    if (maxPartSize < desc.partsToRanks[i]->size) { maxPartSize = desc.partsToRanks[i]->size; }
    if (minPartSize > desc.partsToRanks[i]->size) { minPartSize = desc.partsToRanks[i]->size; }
  }

  ASSERT(desc.M >= desc.N,
         "MLCommon::LinAlg::opg Number of rows of input"
         " matrix can not be less than number of columns");
  ASSERT(minPartSize >= desc.N,
         "MLCommon::LinAlg::opg Number of rows in "
         " any part of matrix can not be less than number of columns in the "
         "matrix");
  ASSERT(desc.layout == Matrix::Layout::LayoutColMajor,
         "Multi::QRDecomp: Intra block layout other than column major is not"
         " supported.");

  T* r1Parts     = nullptr;
  T* r1Collected = nullptr;
  T* r1Corrected = nullptr;
  T* q2          = nullptr;
  T* r2          = nullptr;
  T* q2RowMajor  = nullptr;
  T* q2Parts     = nullptr;

  // We reuse these two for multiple temporary buffers
  rmm::device_uvector<T> aBuffer(numParts * N * N, userStream);
  rmm::device_uvector<T> bBuffer(numParts * N * N, userStream);

  // Buffer for holding GEMM result
  rmm::device_uvector<T> gemmBuffer(maxPartSize * N, userStream);

  r1Parts = aBuffer.data();
  q2Parts = bBuffer.data();

  if (myRank == 0) {
    r1Collected = bBuffer.data();
    r1Corrected = aBuffer.data();
    q2          = bBuffer.data();
    q2RowMajor  = aBuffer.data();
  }

  if (myRank == 0) {
    // At rank 0, r2 can directly be captured in outR
    r2 = outR;
  } else {
    // At other ranks, r2 needs to be captured in a separate buffer before final
    // upper triangular copy
    r2 = aBuffer.data();
  }
  // number of rows or column in any part can not be higher than 2^31 because
  // single gpu QR assumes int data type for rows. columns
  RAFT_CUDA_TRY(cudaMemsetAsync(r1Parts, 0, numLocalParts * N * N * sizeof(T), userStream));

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      cudaStream_t stream = h.get_next_usable_stream();

      raft::linalg::qrGetQR(h,
                            inParts[localId]->ptr,
                            outQParts[localId]->ptr,
                            r1Parts + localId * N * N,
                            partSizes[i],
                            N,
                            stream);
      localId++;
    }
  }
  h.sync_stream_pool();
  comm.sync_stream(userStream);

  std::vector<raft::comms::request_t> requests;
  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      // Send the part
      requests.resize(requests.size() + 1);
      comm.isend(r1Parts + localId * N * N, N * N, 0, 0, &requests.back());
      localId++;
    }
    if (myRank == 0) {
      // Receive the part
      requests.resize(requests.size() + 1);
      comm.irecv(r1Collected + i * N * N, N * N, desc.partsToRanks[i]->rank, 0, &requests.back());
    }
  }
  comm.waitall(requests.size(), requests.data());
  requests.clear();
  if (myRank == 0) {
    dim3 block(256);
    dim3 grid((numParts * N * N + block.x - 1) / block.x);
    r1Repack<<<grid, block, 0, userStream>>>(
      r1Collected, r1Corrected, numParts * N * N, N, numParts);
    raft::linalg::qrGetQR(h, r1Corrected, q2, r2, numParts * N, N, userStream);
    raft::linalg::transpose(h, q2, q2RowMajor, numParts * N, N, userStream);
  }
  comm.sync_stream(userStream);

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (myRank == 0) {
      // Send the Part
      requests.resize(requests.size() + 1);
      comm.isend(q2RowMajor + i * N * N, N * N, desc.partsToRanks[i]->rank, 0, &requests.back());
    }
    if (desc.partsToRanks[i]->rank == myRank) {
      // Receive the part
      requests.resize(requests.size() + 1);
      comm.irecv(q2Parts + localId * N * N, N * N, 0, 0, &requests.back());
      localId++;
    }
  }
  comm.waitall(requests.size(), requests.data());
  requests.clear();

  for (int i = 0, localId = 0; i < numParts; i++) {
    if (desc.partsToRanks[i]->rank == myRank) {
      cudaStream_t stream = h.get_next_usable_stream();

      raft::linalg::gemm(h,
                         gemmBuffer.data(),
                         outQParts[localId]->ptr,
                         q2Parts + localId * N * N,
                         partSizes[i],
                         N,
                         N,
                         true,
                         true,
                         false,
                         stream);
      RAFT_CUDA_TRY(cudaMemcpyAsync(outQParts[localId]->ptr,
                                    gemmBuffer.data(),
                                    partSizes[i] * N * sizeof(T),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
      localId++;
    }
  }
  h.sync_stream_pool();

  comm.bcast(r2, N * N, 0, userStream);
  comm.sync_stream(userStream);

  dim3 block(128, 4, 1);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y, 1);
  copyUpperTriangle<<<1, 10, 0, userStream>>>(outR, r2, N);
  RAFT_CUDA_TRY(cudaPeekAtLastError());  // To capture launch related errors
}

// Instantiations
void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& outQParts,
              float* outR,
              std::vector<Matrix::Data<float>*>& inParts,
              Matrix::PartDescriptor& desc,
              int myRank)
{
  qrDecomp<float>(h, outQParts, outR, inParts, desc, myRank);
}

void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& outQParts,
              double* outR,
              std::vector<Matrix::Data<double>*>& inParts,
              Matrix::PartDescriptor& desc,
              int myRank)
{
  qrDecomp<double>(h, outQParts, outR, inParts, desc, myRank);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
