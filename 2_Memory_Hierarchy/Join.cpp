#include "JoinUtils.hpp"
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
using namespace std;


int splitTitle(const vector<TitleRelation>& titleRelation, int index_of_cutoff) {
    if (index_of_cutoff < 0 || index_of_cutoff >= static_cast<int>(titleRelation.size()))
        return 0;

    int current_id = titleRelation[index_of_cutoff].titleId;
    int current_index = index_of_cutoff;

    if (current_index + 1 < static_cast<int>(titleRelation.size()) &&
        titleRelation[current_index + 1].titleId != current_id) {
        while (current_index > 0 && titleRelation[current_index - 1].titleId == current_id) {
            current_index--;
        }
    }

    return current_index;
}

// Safely walks backward through titleRelation to find cutoff
int backTitle(const vector<TitleRelation>& titleRelation, int movieId, int index_of_cutoff) {
    int current_index = index_of_cutoff;
    while (current_index > 0 && titleRelation[current_index].titleId > movieId) {
        current_index--;
    }
    return current_index;
}

// Safely splits castRelation vector at cutoff index
int splitCast(const vector<CastRelation>& castRelation, int index_of_cutoff) {
    if (index_of_cutoff < 0 || index_of_cutoff >= static_cast<int>(castRelation.size()))
        return 0;

    int current_id = castRelation[index_of_cutoff].movieId;
    int current_index = index_of_cutoff;

    if (current_index + 1 < static_cast<int>(castRelation.size()) &&
        castRelation[current_index + 1].movieId != current_id) {
        while (current_index > 0 && castRelation[current_index - 1].movieId == current_id) {
            current_index--;
        }
    }

    return current_index;
}

// Safely walks backward through castRelation to find cutoff
int backCast(const vector<CastRelation>& castRelation, int titleId, int index_of_cutoff) {
    int current_index = index_of_cutoff;
    while (current_index > 0 && castRelation[current_index].movieId > titleId) {
        current_index--;
    }
    return current_index;
}

// Determines how to slice castRelation and titleRelation at a cutoff point
vector<int> splitRelations(const vector<CastRelation>& castRelation, const vector<TitleRelation>& titleRelation, int cast_index_of_cutoff, int title_index_of_cutoff) {
    if (castRelation.empty() || titleRelation.empty() ||
        cast_index_of_cutoff >= static_cast<int>(castRelation.size()) ||
        title_index_of_cutoff >= static_cast<int>(titleRelation.size())) {
        return {0, 0};
        }

    int title_id = titleRelation[title_index_of_cutoff].titleId;
    int cast_id = castRelation[cast_index_of_cutoff].movieId;

    if (title_id > cast_id) {
        int title_index = splitTitle(titleRelation, title_index_of_cutoff);
        int cast_index = backCast(castRelation, titleRelation[title_index].titleId, cast_index_of_cutoff);
        return {static_cast<int>(title_index), static_cast<int>(cast_index)};
    } else {
        int cast_index = splitCast(castRelation, cast_index_of_cutoff);
        int title_index = backTitle(titleRelation, castRelation[cast_index].movieId, title_index_of_cutoff);
        return {static_cast<int>(title_index), static_cast<int>(cast_index)};
    }
}

// Performs join on two slices of cast/title relation
vector<ResultRelation> performJoinThread(const vector<CastRelation>& castRelation, const vector<TitleRelation>& titleRelation) {
    vector<ResultRelation> resultTuples;
    int pointer_cast = 0;
    int pointer_title = 0;
    int old_position = 0;

    while (pointer_cast < static_cast<int>(castRelation.size()) && pointer_title < static_cast<int> (titleRelation.size())) {
        if (castRelation[pointer_cast].movieId < titleRelation[pointer_title].titleId) {
            pointer_cast++;
        } else if (castRelation[pointer_cast].movieId > titleRelation[pointer_title].titleId) {
            pointer_title++;
        } else {
            old_position = pointer_cast;
            while (pointer_cast < castRelation.size() &&
                   castRelation[pointer_cast].movieId == titleRelation[pointer_title].titleId) {
                resultTuples.push_back(createResultTuple(castRelation[pointer_cast], titleRelation[pointer_title]));
                pointer_cast++;
            }
            pointer_cast = old_position;
            pointer_title++;
        }
    }

    return resultTuples;
}

vector<ResultRelation> performJoin(const vector<CastRelation>& castRelation, const vector<TitleRelation>& titleRelation, int numThreads) {
    int half_cache_size_with_padding = 256 * 1024;

    if (castRelation.empty()) {
        printf("Size is empty!");
        return {};
    }
    int index_of_cutoff = half_cache_size_with_padding / static_cast<int>(sizeof(castRelation[0]));

    vector<vector<CastRelation>> castSlices;
    vector<vector<TitleRelation>> titleSlices;
    vector<ResultRelation> resultRelation;

    int title_offset = 0;
    int cast_offset = 0;
    while (castRelation.size() > cast_offset + index_of_cutoff &&
           titleRelation.size() > title_offset + index_of_cutoff) {

        vector<int> splitIndices = splitRelations(castRelation, titleRelation, index_of_cutoff + cast_offset, index_of_cutoff + title_offset);
        int title_cutoff = splitIndices[0];
        int cast_cutoff  = splitIndices[1];

        auto cast_end = std::min(castRelation.size(), static_cast<size_t>(cast_cutoff));
        auto title_end = std::min(titleRelation.size(), static_cast<size_t>(title_cutoff));

        castSlices.emplace_back(castRelation.begin() + cast_offset, castRelation.begin() + cast_end);
        titleSlices.emplace_back(titleRelation.begin() + title_offset, titleRelation.begin() + title_end);

        title_offset += title_cutoff;
        cast_offset += cast_cutoff;
    }

    if (cast_offset < castRelation.size() && title_offset < titleRelation.size()) {
        castSlices.emplace_back(castRelation.begin() + cast_offset, castRelation.end());
        titleSlices.emplace_back(titleRelation.begin() + title_offset, titleRelation.end());
    }

    if(castSlices.size() != titleSlices.size()) {
        printf("Unterschiedlich viele Chunks!");
        return {};
    }

    vector<vector<ResultRelation>> thread_results(castSlices.size());

    for (int i = 0; i < thread_results.size(); i++) {
        auto estimatedResultCount = castSlices[i].size() * 1.25;
        thread_results[i].reserve(estimatedResultCount);
    }

#pragma omp parallel for schedule(dynamic) num_threads(numThreads) default(none) shared(castSlices, titleSlices, thread_results)
    for (int i = 0; i < static_cast<int>(castSlices.size()); ++i) {
        thread_results[i] = performJoinThread(castSlices[i], titleSlices[i]);
    }

    size_t totalSize = 0;
    for (const auto& localResultRelation : thread_results) {
        totalSize += localResultRelation.size();
    }

    resultRelation.reserve(totalSize);

    for (const auto& vec : thread_results) {
        resultRelation.insert(resultRelation.end(), vec.begin(), vec.end());
    }

    return resultRelation;
}
