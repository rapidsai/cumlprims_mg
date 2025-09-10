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
 * \brief    Multi-GPU QR decomposition for tall (rows > columns) matrices.
 *
 * This function implements multi-GPU QR decomposition for tall matrices
 * descibed in https://arxiv.org/abs/1301.1071.
 * \param       h               cuML handle object.
 * \param[out]  outQParts       Factor Q of input matrix QR decomposition. Q
                                matrix is distributed among ranks according to
                                desc description. The size of Q matrix is
                                same as size of 'in' matrix, i.e. M x N.
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

void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& outQParts,
              float* outR,
              std::vector<Matrix::Data<float>*>& inXParts,
              Matrix::PartDescriptor& desc,
              int myRank);

void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& outQParts,
              double* outR,
              std::vector<Matrix::Data<double>*>& inXParts,
              Matrix::PartDescriptor& desc,
              int myRank);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
