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

#include <cumlprims/opg/linalg/norm.hpp>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/sum.cuh>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG var calculation.
 * @output param Matrix: var of every column of the "in"
 * @input param in: all the data partitions
 * @input param inDesc: MNMG description of the input data
 * @input param comm: communicator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 */
template <typename T, int TPB = 256>
void colNorm2NoSeq_impl(const raft::handle_t& handle,
                        Matrix::Data<T>& out,
                        const std::vector<Matrix::Data<T>*>& in,
                        const Matrix::PartDescriptor& inDesc,
                        cudaStream_t* streams,
                        int n_streams)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<T> local_means_tmp(in.size() * inDesc.N, streams[0]);
  rmm::device_uvector<T> local_means_tmp_t(in.size() * inDesc.N, streams[0]);

  std::vector<Matrix::RankSizePair*> localBlocks = inDesc.blocksOwnedBy(comm.get_rank());

  for (int i = 0; i < localBlocks.size(); i++) {
    T* loc = local_means_tmp.data() + (i * inDesc.N);
    raft::linalg::colNorm<raft::linalg::L2Norm, false>(loc,
                          in[i]->ptr,
                          inDesc.N,
                          localBlocks[i]->size,
                          streams[i % n_streams],
                          [] __device__(T v) { return v; });
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }

  raft::linalg::transpose(handle,
                          local_means_tmp.data(),
                          local_means_tmp_t.data(),
                          inDesc.N,
                          localBlocks.size(),
                          streams[0]);

  raft::stats::sum<false>(local_means_tmp.data(),
                   local_means_tmp_t.data(),
                   inDesc.N,
                   localBlocks.size(),
                   streams[0]);

  comm.allreduce(local_means_tmp.data(), out.ptr, inDesc.N, raft::comms::op_t::SUM, streams[0]);

  comm.sync_stream(streams[0]);
}

/**
 * @brief performs MNMG var calculation.
 * @output param Matrix: var of every column of the "in"
 * @input param in: all the data partitions
 * @input param inDesc: MNMG description of the input data
 * @input param comm: communicator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 */
template <typename T, int TPB = 256>
void colNorm2_impl(const raft::handle_t& handle,
                   Matrix::Data<T>& out,
                   const std::vector<Matrix::Data<T>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams)
{
  colNorm2NoSeq_impl(handle, out, in, inDesc, streams, n_streams);

  raft::matrix::seqRoot(out.ptr, out.ptr, T(1), inDesc.N, streams[0], false);
}

void colNorm2(const raft::handle_t& handle,
              Matrix::Data<double>& out,
              const std::vector<Matrix::Data<double>*>& in,
              const Matrix::PartDescriptor& inDesc,
              cudaStream_t* streams,
              int n_streams)
{
  colNorm2_impl<double>(handle, out, in, inDesc, streams, n_streams);
}

void colNorm2(const raft::handle_t& handle,
              Matrix::Data<float>& out,
              const std::vector<Matrix::Data<float>*>& in,
              const Matrix::PartDescriptor& inDesc,
              cudaStream_t* streams,
              int n_streams)
{
  colNorm2_impl<float>(handle, out, in, inDesc, streams, n_streams);
}

void colNorm2NoSeq(const raft::handle_t& handle,
                   Matrix::Data<double>& out,
                   const std::vector<Matrix::Data<double>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams)
{
  colNorm2NoSeq_impl<double>(handle, out, in, inDesc, streams, n_streams);
}

void colNorm2NoSeq(const raft::handle_t& handle,
                   Matrix::Data<float>& out,
                   const std::vector<Matrix::Data<float>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams)
{
  colNorm2NoSeq_impl<float>(handle, out, in, inDesc, streams, n_streams);
}

};  // namespace opg
// end namespace opg
};  // namespace LinAlg
// end namespace LinAlg
};  // namespace MLCommon
// end namespace MLCommon
