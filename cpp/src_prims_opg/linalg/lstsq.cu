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

#include <cumlprims/opg/linalg/lstsq.hpp>
#include <cumlprims/opg/linalg/mv_aTb.hpp>
#include <cumlprims/opg/linalg/svd.hpp>
#include <raft/linalg/gemv.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <rmm/device_uvector.hpp>

#include "comm_utils.h"

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG Least squares calculation.
 */
template <typename T>
void lstsqEig_impl(const raft::handle_t& handle,
                   const std::vector<Matrix::Data<T>*>& A,
                   const Matrix::PartDescriptor& ADesc,
                   const std::vector<Matrix::Data<T>*>& b,
                   T* w,
                   cudaStream_t* streams,
                   int n_streams)
{
  auto& comm = handle.get_comms();
  int rank   = comm.get_rank();

  rmm::device_uvector<T> S(ADesc.N, streams[0]);
  rmm::device_uvector<T> V(ADesc.N * ADesc.N, streams[0]);
  std::vector<Matrix::Data<T>*> U;
  std::vector<Matrix::Data<T>> U_temp;

  std::vector<Matrix::RankSizePair*> partsToRanks = ADesc.blocksOwnedBy(rank);
  size_t total_size                               = 0;

  for (int i = 0; i < partsToRanks.size(); i++) {
    total_size += partsToRanks[i]->size;
  }
  total_size = total_size * ADesc.N;

  rmm::device_uvector<T> U_parts(total_size, streams[0]);
  T* curr_ptr = U_parts.data();

  for (int i = 0; i < partsToRanks.size(); i++) {
    Matrix::Data<T> d;
    d.totalSize = partsToRanks[i]->size;
    d.ptr       = curr_ptr;
    curr_ptr    = curr_ptr + (partsToRanks[i]->size * ADesc.N);
    U_temp.push_back(d);
  }

  for (int i = 0; i < A.size(); i++) {
    U.push_back(&(U_temp[i]));
  }

  svdEig(handle, A, ADesc, U, S.data(), V.data(), streams, n_streams);

  // we use a temporary vector to avoid doing re-using w in the last step, the
  // gemv, which could cause a very sporadic race condition in Pascal and
  // Turing GPUs that caused it to give the wrong results. Details:
  // https://github.com/rapidsai/cuml/issues/1739
  rmm::device_uvector<T> tmp_vector(ADesc.N, streams[0]);

  Matrix::Data<T> w_out;
  w_out.ptr       = tmp_vector.data();
  w_out.totalSize = ADesc.N;

  mv_aTb(handle, w_out, U, ADesc, b, streams, n_streams);

  raft::matrix::matrixVectorBinaryDivSkipZero<false, true>(
    tmp_vector.data(), S.data(), size_t(1), ADesc.N, streams[0]);

  raft::linalg::gemv(handle, V.data(), ADesc.N, ADesc.N, tmp_vector.data(), w, false, streams[0]);
}

void lstsqEig(const raft::handle_t& handle,
              const std::vector<Matrix::Data<float>*>& A,
              const Matrix::PartDescriptor& ADesc,
              const std::vector<Matrix::Data<float>*>& b,
              float* w,
              cudaStream_t* streams,
              int n_streams)
{
  lstsqEig_impl(handle, A, ADesc, b, w, streams, n_streams);
}

void lstsqEig(const raft::handle_t& handle,
              const std::vector<Matrix::Data<double>*>& A,
              const Matrix::PartDescriptor& ADesc,
              const std::vector<Matrix::Data<double>*>& b,
              double* w,
              cudaStream_t* streams,
              int n_streams)
{
  lstsqEig_impl(handle, A, ADesc, b, w, streams, n_streams);
}

};  // namespace opg
// end namespace opg
};  // namespace LinAlg
// end namespace LinAlg
};  // namespace MLCommon
// end namespace MLCommon
