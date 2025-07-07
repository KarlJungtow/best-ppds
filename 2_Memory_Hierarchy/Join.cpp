#include "JoinUtils.hpp"
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
using namespace std;

// Finds the start of the current group
uint_fast32_t findGroupStart(const vector<CastRelation>& relation, uint_fast32_t pos) {
    const int32_t current_id = relation[pos].movieId;
    while (pos > 0 && relation[pos-1].movieId == current_id) {
        --pos;
    }
    return pos;
}

// Finds the end of the current group (exclusive)
uint_fast32_t findGroupEnd(const vector<CastRelation>& relation, uint_fast32_t pos) {
    const int32_t current_id = relation[pos].movieId;
    const size_t size = relation.size();
    while (pos < size && relation[pos].movieId == current_id) {
        ++pos;
    }
    return pos;
}

// Finds the start of the current group
uint_fast32_t findGroupStart(const vector<TitleRelation>& relation, uint_fast32_t pos) {
    const int32_t current_id = relation[pos].titleId;
    while (pos > 0 && relation[pos-1].titleId == current_id) {
        --pos;
    }
    return pos;
}

// Finds the end of the current group (exclusive)
uint_fast32_t findGroupEnd(const vector<TitleRelation>& relation, uint_fast32_t pos) {
    const int32_t current_id = relation[pos].titleId;
    const size_t size = relation.size();
    while (pos < size && relation[pos].titleId == current_id) {
        ++pos;
    }
    return pos;
}

// Performs join on two slices using group-aware merge
vector<ResultRelation> performJoinThread(const vector<CastRelation>& castRelation,
                                         const vector<TitleRelation>& titleRelation) {
    vector<ResultRelation> resultTuples;
    size_t c = 0;
    size_t t = 0;
    const size_t cast_size = castRelation.size();
    const size_t title_size = titleRelation.size();

    while (c < cast_size && t < title_size) {
        const int32_t cast_id = castRelation[c].movieId;
        const int32_t title_id = titleRelation[t].titleId;

        if (cast_id < title_id) {
            // Advance to next cast group
            c = findGroupEnd(castRelation, c);
        } else if (cast_id > title_id) {
            // Advance to next title group
            t = findGroupEnd(titleRelation, t);
        } else {
            // Found matching groups
            const size_t cast_group_start = c;
            const size_t cast_group_end = findGroupEnd(castRelation, c);
            const size_t title_group_start = t;
            const size_t title_group_end = findGroupEnd(titleRelation, t);

            // Cross product of matching groups
            for (size_t i = cast_group_start; i < cast_group_end; ++i) {
                for (size_t j = title_group_start; j < title_group_end; ++j) {
                    resultTuples.push_back(createResultTuple(castRelation[i], titleRelation[j]));
                }
            }

            // Move to next groups
            c = cast_group_end;
            t = title_group_end;
        }
    }

    return resultTuples;
}

// Creates cache-sized chunks respecting group boundaries
vector<ResultRelation> performJoin(const vector<CastRelation>& castRelation,
                                   const vector<TitleRelation>& titleRelation,
                                   int numThreads) {
    const size_t half_cache_bytes = 256 * 1024;  // 256KB
    const size_t min_chunk_size = half_cache_bytes / sizeof(CastRelation);

    if (castRelation.empty() || titleRelation.empty()) {
        return {};
    }

    vector<vector<CastRelation>> castSlices;
    vector<vector<TitleRelation>> titleSlices;
    size_t c_pos = 0;
    size_t t_pos = 0;

    // Create chunks respecting group boundaries
    while (c_pos < castRelation.size() && t_pos < titleRelation.size()) {
        // Start new chunks at current positions
        const size_t c_start = c_pos;
        const size_t t_start = t_pos;

        // Determine chunk end positions
        size_t c_end = min(c_pos + min_chunk_size, castRelation.size());
        size_t t_end = min(t_pos + min_chunk_size, titleRelation.size());

        // Extend to end of current groups
        if (c_end < castRelation.size()) {
            c_end = findGroupEnd(castRelation, c_end - 1);
        }
        if (t_end < titleRelation.size()) {
            t_end = findGroupEnd(titleRelation, t_end - 1);
        }

        // Create slices
        castSlices.emplace_back(
            castRelation.begin() + c_start,
            castRelation.begin() + c_end
        );

        titleSlices.emplace_back(
            titleRelation.begin() + t_start,
            titleRelation.begin() + t_end
        );

        // Move to next chunks
        c_pos = c_end;
        t_pos = t_end;
    }

    vector<vector<ResultRelation>> thread_results(castSlices.size());

    #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
    for (int i = 0; i < static_cast<int>(castSlices.size()); ++i) {
        thread_results[i] = performJoinThread(castSlices[i], titleSlices[i]);
    }

    // Combine results
    vector<ResultRelation> resultRelation;
    size_t totalSize = 0;
    for (const auto& res : thread_results) {
        totalSize += res.size();
    }
    resultRelation.reserve(totalSize);
    for (const auto& vec : thread_results) {
        resultRelation.insert(resultRelation.end(), vec.begin(), vec.end());
    }

    return resultRelation;
}