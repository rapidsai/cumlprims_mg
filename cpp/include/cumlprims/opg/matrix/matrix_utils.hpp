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
#include <raft/random/rng.cuh>

namespace MLCommon {
namespace Matrix {
namespace opg {

void gatherPart(const raft::handle_t& h,
                float* gatheredPart,
                std::vector<Matrix::Data<float>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream);

void allGatherPart(const raft::handle_t& h,
                   float* gatheredPart,
                   std::vector<Matrix::Data<float>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream);
void gather(const raft::handle_t& h,
            float* gatheredMatrix,
            std::vector<Matrix::Data<float>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream);

void allGather(const raft::handle_t& h,
               float* gatheredMatrix,
               std::vector<Matrix::Data<float>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream);

void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<float>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream);

void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<float>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               float low  = -1.0f,
               float high = 1.0f);

void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<float>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void printRaw2D(float* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream);

void print(const raft::handle_t& h,
           std::vector<Matrix::Data<float>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream);

//------------------------------------------------------------------------------

void gatherPart(const raft::handle_t& h,
                double* gatheredPart,
                std::vector<Matrix::Data<double>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream);

void allGatherPart(const raft::handle_t& h,
                   double* gatheredPart,
                   std::vector<Matrix::Data<double>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream);

void gather(const raft::handle_t& h,
            double* gatheredMatrix,
            std::vector<Matrix::Data<double>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream);

void allGather(const raft::handle_t& h,
               double* gatheredMatrix,
               std::vector<Matrix::Data<double>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream);

void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<double>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream);

void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<double>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               double low  = -1.0,
               double high = 1.0);

void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<double>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void printRaw2D(double* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream);

void print(const raft::handle_t& h,
           std::vector<Matrix::Data<double>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream);
}  // end namespace opg
}  // namespace Matrix
}  // end namespace MLCommon
