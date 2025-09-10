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
namespace Stats {
namespace opg {

/**
 * @brief performs MNMG covariance calculation.
 * @output param covar: resulting covariance matrix
 * @input param data: the data that cov matrix is calculated for
 * @input param dataDesc: MNMG description of the input data
 * @input param mu: mean of every column in data
 * @input param comm: communicator
 * @input param allocator: data allocator
 * @input param streams: cuda streams
 * @input param n_streams: number of streams
 * @input param handle: cublas handle
 */
void cov(const raft::handle_t& handle,
         Matrix::Data<float>& covar,
         const std::vector<Matrix::Data<float>*>& data,
         const Matrix::PartDescriptor& dataDesc,
         Matrix::Data<float>& mu,
         bool sample,
         cudaStream_t* streams,
         int n_streams);

void cov(const raft::handle_t& handle,
         Matrix::Data<double>& covar,
         const std::vector<Matrix::Data<double>*>& data,
         const Matrix::PartDescriptor& dataDesc,
         Matrix::Data<double>& mu,
         bool sample,
         cudaStream_t* streams,
         int n_streams);

/** @} */

}  // end namespace opg
}  // end namespace Stats
}  // end namespace MLCommon
