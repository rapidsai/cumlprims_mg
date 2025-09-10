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

#include <stdint.h>

#include <cumlprims/opg/matrix/data.hpp>
#include <ostream>
#include <set>
#include <vector>

namespace MLCommon {
namespace Matrix {

/** Describes the data layout */
enum Layout {
  /** row major layout */
  LayoutRowMajor = 0,
  /** column major layout */
  LayoutColMajor
};

struct RankSizePair {
  RankSizePair() : rank(-1), size(0) {}

  RankSizePair(int _rank, size_t _size) : rank(_rank), size(_size) {}

  int rank;

  /**
   * Total number of rows
   */
  size_t size;
};

struct PartDescriptor {
  /** total number of rows */
  size_t M;
  /** total number of columns */
  size_t N;

  int rank;

  Layout layout;
  /** mapping of each block (in col-major order) to the device that owns it */
  std::vector<RankSizePair*> partsToRanks;

  /**
   * @brief For a given matrix and block-sizes construct the corresponding
   *  descriptor for it. This is useful when we are dealing with standard
   *  row/column-wise block-cyclic data distribution, as seen in other popular
   *  multi-node packages like magma etc.
   * @param _M total number of rows of this matrix
   * @param _N total number of columns
   * @param _partsToRanks mapping of ranks to parts and sizes
   */
  PartDescriptor(size_t _M,
                 size_t _N,
                 const std::vector<RankSizePair*>& _partsToRanks,
                 int rank,
                 Layout _layout = LayoutColMajor);

  /** total number of blocks across all workers */
  int totalBlocks() const { return partsToRanks.size(); }

  /** Count the total number of blocks owned by a given rank */
  int totalBlocksOwnedBy(int rank) const;

  std::set<int> uniqueRanks();

  std::vector<size_t> startIndices() const;

  std::vector<size_t> startIndices(int rank) const;

  /**
   * @brief Returns the vector of blocks (each identified by linearBLockIndex)
   * owned by the given rank
   */
  std::vector<RankSizePair*> blocksOwnedBy(int rank) const;

  /** Count the total number of matrix elements owned by a given rank */
  size_t totalElementsOwnedBy(int rank) const;

  friend std::ostream& operator<<(std::ostream& os, const PartDescriptor& desc);
  friend bool operator==(const PartDescriptor& a, const PartDescriptor& b);
};

/** Print matrix descriptor in human readable form */
std::ostream& operator<<(std::ostream& os, const PartDescriptor& desc);

/** compare 2 descriptor objects */
bool operator==(const PartDescriptor& a, const PartDescriptor& b);

};  // end namespace Matrix
};  // end namespace MLCommon
