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
#include <cumlprims/opg/matrix/matrix_utils.hpp>
#include <raft/linalg/eig.cuh>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * \brief    Multi-GPU version of eigen decomposition. This function works for
 *           symmetric matrices only. Whole input matrix is gathered at rank 0
 *           and Eigen decomposition is carried out sequentially.
 *
 * \tparam      T               Data-type for in, eigen values and eigen
 * vectors. \param       h               cuML handle object. \param[out]
 * eigenValues     Output N Eigen values. \param[out]  eigenVectors    Output N
 * Eigen vectors of size N x 1. \param[in]   in              Input symmetric
 * matrix of size N x N. \param[in]   desc            Descriptor of input matrix
 * in.
 */

template <typename T>
void eigDC(const raft::handle_t& h,
           T* eigenValues,
           T* eigenVectors,
           std::vector<Matrix::Data<T>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  ASSERT(desc.N == desc.M,
         "MLCommon::LinAlg::opg:Eig: Matrix needs to be square for Eigen"
         " computation");

  const auto& comm = h.get_comms();

  rmm::device_uvector<T> inGathered(0, stream);
  if (myRank == 0) { inGathered.resize(desc.M * desc.N, stream); }
  Matrix::opg::gather(h, inGathered.data(), inParts, desc, 0, myRank, stream);

  if (myRank == 0) {
    raft::linalg::eigDC(h, inGathered.data(), desc.N, desc.N, eigenVectors, eigenValues, stream);
  }

  comm.bcast(eigenVectors, desc.N * desc.N, 0, stream);
  comm.bcast(eigenValues, desc.N, 0, stream);
}

template <typename T>
void eigJacobi(const raft::handle_t& h,
               T* eigenValues,
               T* eigenVectors,
               std::vector<Matrix::Data<T>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  const auto& comm = h.get_comms();

  rmm::device_uvector<T> inGathered(0, stream);
  if (myRank == 0) { inGathered.resize(desc.N * desc.N, stream); }
  Matrix::opg::gather(h, inGathered.data(), inParts, desc, 0, myRank, stream);

  if (myRank == 0) {
    raft::linalg::eigJacobi(
      h, inGathered.data(), desc.N, desc.N, eigenVectors, eigenValues, stream);
  }

  comm.bcast(eigenVectors, desc.N * desc.N, 0, stream);
  comm.bcast(eigenValues, desc.N, 0, stream);
}

// Instantiations
void eigDC(const raft::handle_t& h,
           float* eigenValues,
           float* eigenVectors,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  eigDC<float>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

void eigDC(const raft::handle_t& h,
           double* eigenValues,
           double* eigenVectors,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  eigDC<double>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

void eigJacobi(const raft::handle_t& h,
               float* eigenValues,
               float* eigenVectors,
               std::vector<Matrix::Data<float>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  eigJacobi<float>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

void eigJacobi(const raft::handle_t& h,
               double* eigenValues,
               double* eigenVectors,
               std::vector<Matrix::Data<double>*>& inParts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  eigJacobi<double>(h, eigenValues, eigenVectors, inParts, desc, myRank, stream);
}

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
