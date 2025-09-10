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
 * \brief    Multi-GPU version of Eigen decomposition. This function works for
 *           symmetric matrices only. Whole input matrix is gathered at rank 0
 *           and Eigen decomposition is carried out sequentially.
 *
 * \param       h               cuML handle object.
 * \param[out]  eigenValues     Output N Eigen values.
 * \param[out]  eigenVectors    Output N Eigen vectors of size N x 1.
 * \param[in]   in              Input symmetric matrix of size N x N.
 * \param[in]   desc            Descriptor of input matrix in.
 */

void eigDC(const raft::handle_t& h,
           float* eigenValues,
           float* eigenVectors,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void eigDC(const raft::handle_t& h,
           double* eigenValues,
           double* eigenVectors,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void eigJacobi(const raft::handle_t& h,
               float* eigenValues,
               float* eigenVectors,
               std::vector<Matrix::Data<float>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

void eigJacobi(const raft::handle_t& h,
               double* eigenValues,
               double* eigenVectors,
               std::vector<Matrix::Data<double>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
