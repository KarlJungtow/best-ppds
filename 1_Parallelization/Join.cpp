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

#include "TimerUtil.hpp"
#include "JoinUtils.hpp"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <gtest/gtest.h>
#include <omp.h>


std::vector<ResultRelation> performJoin(const std::vector<CastRelation>& castRelation,
                                        const std::vector<TitleRelation>& titleRelation,
                                        int numThreads) {
    omp_set_num_threads(numThreads);
    std::vector<ResultRelation> resultTuples;
    std::vector<std::unordered_map<int, std::vector<TitleRelation>>> localHashTables(numThreads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localMap = localHashTables[tid];

        size_t chunkSize = (titleRelation.size() + numThreads - 1) / numThreads;
        size_t start = tid * chunkSize;
        size_t end = std::min(start + chunkSize, titleRelation.size());

        for (size_t i = start; i < end; ++i) {
            const auto& title = titleRelation[i];
            localMap[title.titleId].emplace_back(title);
        }
    }

    std::unordered_map<int, std::vector<TitleRelation>> globalHashTable;
    for (const auto& localMap : localHashTables) {
        for (const auto& [key, values] : localMap) {
            globalHashTable[key].insert(globalHashTable[key].end(), values.begin(), values.end());
        }
    }

    std::vector<size_t> matchCounts(castRelation.size());
#pragma omp parallel for
    for (size_t i = 0; i < castRelation.size(); ++i) {
        int movieId = castRelation[i].movieId;
        auto it = globalHashTable.find(movieId);
        matchCounts[i] = (it != globalHashTable.end()) ? it->second.size() : 0;
    }

    std::vector<size_t> offsets(castRelation.size() + 1, 0);
    for (size_t i = 0; i < castRelation.size(); ++i) {
        offsets[i + 1] = offsets[i] + matchCounts[i];
    }

    resultTuples.resize(offsets.back());

#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < castRelation.size(); ++i) {
        const auto& castTuple = castRelation[i];
        int movieId = castTuple.movieId;
        size_t outputPos = offsets[i];

        auto it = globalHashTable.find(movieId);
        if (it != globalHashTable.end()) {
            for (const auto& titleTuple : it->second) {
                resultTuples[outputPos++] = createResultTuple(castTuple, titleTuple);
            }
        }
    }

    return resultTuples;
}