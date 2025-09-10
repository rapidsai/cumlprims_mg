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
#include <raft/core/handle.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG Least squares calculation.
 */
void lstsqEig(const raft::handle_t& handle,
              const std::vector<Matrix::Data<float>*>& A,
              const Matrix::PartDescriptor& ADesc,
              const std::vector<Matrix::Data<float>*>& b,
              float* w,
              cudaStream_t* streams,
              int n_streams);

void lstsqEig(const raft::handle_t& handle,
              const std::vector<Matrix::Data<double>*>& A,
              const Matrix::PartDescriptor& ADesc,
              const std::vector<Matrix::Data<double>*>& b,
              double* w,
              cudaStream_t* streams,
              int n_streams);
/** @} */

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
