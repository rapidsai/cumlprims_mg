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

#include <raft/core/comms.hpp>
#include <raft/util/cuda_utils.cuh>

namespace MLCommon {
namespace opg {

/** sum-reduce single float numbers across workers */
template <typename T>
void reduce_single_sum(
  T* out, const T* in, const raft::comms::comms_t& comm, cudaStream_t stream, int root = 0)
{
  comm.reduce(in, out, 1, raft::comms::op_t::SUM, root, stream);
  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");
}

/** all-reduce single float numbers across workers in Comm */
template <typename T>
void allreduce_single_sum(T* out,
                          const T* in,
                          const raft::comms::comms_t& comm,
                          cudaStream_t stream)
{
  comm.allreduce(in, out, 1, raft::comms::op_t::SUM, stream);
  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");
}

};  // end namespace opg
};  // end namespace MLCommon
