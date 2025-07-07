/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <gtest/gtest.h>
#include <omp.h>

#include <cstdint>
#include <iostream>
#include <list>
#include <span>
#include <unordered_map>
#include <vector>

#include "JoinUtils.hpp"
#include "TimerUtil.hpp"

using RelA = TitleRelation;
using RelB = CastRelation;

constexpr int RADIX_BITS = 10;
constexpr int RADIX_SIZE = 1 << RADIX_BITS;  // 2^RADIX_BITS partitions
constexpr int RADIX_MASK = RADIX_SIZE - 1;

thread_local std::vector<ResultRelation> threadLocalResults;

void radixPartitionMovie(const std::vector<CastRelation> &rel, std::vector<CastRelation> &resRel,
                    std::vector<int32_t> &boarders) {
  boarders.resize(RADIX_SIZE + 1);
  resRel.resize(rel.size());

  std::vector<int32_t> tmpRadixLengths(RADIX_SIZE);
  for (const auto &elm : rel) {
    ++tmpRadixLengths[elm.movieId & RADIX_MASK];
  }

  for (int16_t i = 1; i < RADIX_SIZE; ++i) {
    boarders[i] = boarders[i - 1] + tmpRadixLengths[i - 1];
  }

  tmpRadixLengths = boarders;

  for (size_t i = 0; i < rel.size(); ++i) {
    const auto &elm = rel[i];
    const int32_t numPartition = elm.movieId & RADIX_MASK;
    resRel[tmpRadixLengths[numPartition]] = elm;
    ++tmpRadixLengths[numPartition];
  }

  boarders[boarders.size() - 1] = rel.size();
}

void radixPartitionTitle(const std::vector<TitleRelation> &rel, std::vector<TitleRelation> &resRel,
                    std::vector<int32_t> &boarders) {
  boarders.resize(RADIX_SIZE + 1);
  resRel.resize(rel.size());

  std::vector<int32_t> tmpRadixLengths(RADIX_SIZE);
  for (const auto &elm : rel) {
    ++tmpRadixLengths[elm.titleId & RADIX_MASK];
  }

  for (int16_t i = 1; i < RADIX_SIZE; ++i) {
    boarders[i] = boarders[i - 1] + tmpRadixLengths[i - 1];
  }

  tmpRadixLengths = boarders;

  for (size_t i = 0; i < rel.size(); ++i) {
    const auto &elm = rel[i];
    const int32_t numPartition = elm.titleId & RADIX_MASK;
    resRel[tmpRadixLengths[numPartition]] = elm;
    ++tmpRadixLengths[numPartition];
  }

  boarders[boarders.size() - 1] = rel.size();
}

std::vector<ResultRelation> performJoin(const std::vector<RelB> &relB,
                                        const std::vector<RelA> &relA,
                                        const int numThreads) {
  omp_set_num_threads(numThreads);

  std::vector<int32_t> boardersA{};
  std::vector<int32_t> boardersB{};
  std::vector<RelA> vecPartitionedRelA{};
  std::vector<RelB> vecPartitionedRelB{};

#pragma omp parallel sections
  {
#pragma omp section
    radixPartitionTitle(relA, vecPartitionedRelA, boardersA);
#pragma omp section
    radixPartitionMovie(relB, vecPartitionedRelB, boardersB);
  }

  std::span partitionedRelA(vecPartitionedRelA);
  std::span partitionedRelB(vecPartitionedRelB);
  std::vector<size_t> threadSizes(numThreads);
  std::vector<ResultRelation> resultRelation;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    threadLocalResults.clear();

#pragma omp for
    for (int32_t i = 0; i < RADIX_SIZE; ++i) {
      std::unordered_map<int32_t, std::vector<RelA>> hashTable;
      std::span subSpanA = partitionedRelA.subspan(
          boardersA[i], boardersA[i + 1] - boardersA[i]);
      std::span subSpanB = partitionedRelB.subspan(
          boardersB[i], boardersB[i + 1] - boardersB[i]);
      if (subSpanA.empty() || subSpanB.empty()) {
        continue;
      }
      for (const auto &elm : subSpanA) {
        hashTable[elm.titleId].emplace_back(elm);
      }

      for (const auto &elm : subSpanB) {
        auto found = hashTable.find(elm.movieId);
        if (found == hashTable.end()) continue;

        for (const auto &resElm : found->second) {
          threadLocalResults.emplace_back(createResultTuple(elm, resElm));
        }
      }
    }

    threadSizes[tid] = threadLocalResults.size();

#pragma omp barrier
#pragma omp single
    {
      for (size_t i = 1; i < threadSizes.size(); ++i) {
        threadSizes[i] += threadSizes[i - 1];
      }
      resultRelation.resize(threadSizes.empty() ? 0 : threadSizes.back());
    }

    const size_t offset = (tid == 0) ? 0 : threadSizes[tid - 1];
    std::move(threadLocalResults.begin(), threadLocalResults.end(),
              resultRelation.begin() + offset);
  }
  return resultRelation;
}