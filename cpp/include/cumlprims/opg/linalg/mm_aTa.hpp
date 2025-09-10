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

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG A^T x A calculation.
 * @output param out: resulting A^T x A matrix
 * @input param A: input data
 * @input param ADesc: MNMG description of the A
 * @input param comm: communicator
 * @input param allocator: data allocator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 * @input param handle: cublas handle
 */
void mm_aTa(const raft::handle_t& handle,
            Matrix::Data<double>& out,
            const std::vector<Matrix::Data<double>*>& in1,
            const Matrix::PartDescriptor& in1Desc,
            cudaStream_t* streams,
            int n_streams);

void mm_aTa(const raft::handle_t& handle,
            Matrix::Data<float>& out,
            const std::vector<Matrix::Data<float>*>& in1,
            const Matrix::PartDescriptor& in1Desc,
            cudaStream_t* streams,
            int n_streams);
/** @} */

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
