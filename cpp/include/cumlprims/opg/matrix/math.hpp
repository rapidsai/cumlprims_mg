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

namespace MLCommon {
namespace Matrix {
namespace opg {

template<bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<double>*>& data,
                                   const Matrix::PartDescriptor& inDesc,
                                   const Matrix::Data<double>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams);

template<bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<float>*>& data,
                                   const Matrix::PartDescriptor& inDesc,
                                   const Matrix::Data<float>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams);

template<bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<double>*>& data,
                            const Matrix::PartDescriptor& inDesc,
                            const Matrix::Data<double>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams);

template<bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<float>*>& data,
                            const Matrix::PartDescriptor& inDesc,
                            const Matrix::Data<float>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams);

};  // namespace opg
};  // end namespace Matrix
};  // end namespace MLCommon
