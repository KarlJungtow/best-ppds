#include "JoinUtils.hpp"
#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
using namespace std;


uint_fast32_t splitTitle(const vector<TitleRelation>& titleRelation, int index_of_cutoff) {
    int current_id = titleRelation[index_of_cutoff].titleId;
    int current_index = index_of_cutoff;

    if (titleRelation[current_index+1].titleId != current_id) {
        do {
            current_index--;
        } while (titleRelation[current_index-1].titleId == current_id);
    }
    return current_index;
}

uint_fast32_t backTitle(const vector<TitleRelation>& titleRelation, int movieId, int index_of_cutoff) {
    int current_index = index_of_cutoff;
    while(titleRelation[current_index].titleId > movieId) {
        current_index--;
    }
    return current_index;
}

uint_fast32_t splitCast(const vector<CastRelation>& castRelation, int index_of_cutoff) {
    int current_id = castRelation[index_of_cutoff].movieId;
    int current_index = index_of_cutoff;

    if (castRelation[current_index+1].movieId != current_id) {
        do {
            current_index--;
        } while (castRelation[current_index-1].movieId == current_id);
    }
    return current_index;
}

uint_fast32_t backCast(const vector<CastRelation>& castRelation, int titleId, int index_of_cutoff) {
    int current_index = index_of_cutoff;
    while (castRelation[current_index].movieId > titleId) {
        current_index--;
    }
    return current_index;
}

vector<uint_fast32_t> splitRelations( const vector<CastRelation>& castRelation, const vector<TitleRelation>& titleRelation, int index_of_cutoff) {

    int title_id = titleRelation[index_of_cutoff].titleId;
    int cast_id = castRelation[index_of_cutoff].movieId;

    if (title_id > cast_id) {
        int title_index = splitTitle(titleRelation, index_of_cutoff);
        int cast_index = backCast(castRelation, titleRelation[title_index].titleId, index_of_cutoff);
        return {static_cast<uint_fast32_t>(title_index), static_cast<uint_fast32_t>(cast_index)};
    } else {
        int cast_index = splitCast(castRelation, index_of_cutoff);
        int title_index = backTitle(titleRelation, castRelation[cast_index].movieId, index_of_cutoff);
        return {static_cast<uint_fast32_t>(title_index), static_cast<uint_fast32_t>(cast_index)};
    }
}


vector<ResultRelation> performJoinThread(const vector<CastRelation>& castRelation, const vector<TitleRelation>& titleRelation) {
    // IMPORTANT: You can assume for this benchmark that the join keys are sorted in both relations.
    vector<ResultRelation> resultTuples;

    int pointer_cast = 0;
    int pointer_title = 0;
    int old_position = 0;

    while(pointer_cast < castRelation.size() && pointer_title < titleRelation.size()) {
        if(castRelation[pointer_cast].movieId < titleRelation[pointer_title].titleId) {
            pointer_cast++;
        } else if(castRelation[pointer_cast].movieId > titleRelation[pointer_title].titleId) {
            pointer_title++;
        } else {
            old_position = pointer_cast;
            while(pointer_cast < castRelation.size() && castRelation[pointer_cast].movieId == titleRelation[pointer_title].titleId) {
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
    // L2-Chache Größe
    int half_cache_size_with_padding = 256 * 1024;
    int index_of_cutoff = half_cache_size_with_padding / sizeof(castRelation[0]);

    vector<vector<CastRelation>> castSlices;
    vector<vector<TitleRelation>> titleSlices;
    vector<ResultRelation> resultRelation;

    int offset = 0;
    // Gehe die Relations solange durch, wie beide Größer sind als der halbe Cache
    while (castRelation.size() - offset > half_cache_size_with_padding && titleRelation.size() - offset > half_cache_size_with_padding) {
        vector<uint_fast32_t> splitIndices = splitRelations(castRelation, titleRelation, index_of_cutoff + offset);
        uint_fast32_t title_cutoff = splitIndices[0];
        uint_fast32_t cast_cutoff  = splitIndices[1];

        castSlices.emplace_back(castRelation.begin() + offset, castRelation.begin() + cast_cutoff + offset);
        titleSlices.emplace_back(titleRelation.begin() + offset, titleRelation.begin() + title_cutoff + offset);
        offset += half_cache_size_with_padding;
    }

    // Letztes Stück auch noch hinzufügen
    if (offset < castRelation.size() && offset < titleRelation.size()) {
        castSlices.emplace_back(castRelation.begin() + offset, castRelation.end());
        titleSlices.emplace_back(titleRelation.begin() + offset, titleRelation.end());
    }

    // Vector für lokale Ergebnisse
    vector<vector<ResultRelation>> thread_results(castSlices.size());

    for (size_t i = 0; i < thread_results.size(); i++) {
        size_t estimatedResultCount = castSlices[i].size()*1.25;
        thread_results[i].reserve(estimatedResultCount);
    }


    // Threads einzelne Chunks ausrechnen lassen
    #pragma omp parallel for schedule(dynamic) num_threads(50)
    for (int i = 0; i < static_cast<int>(castSlices.size()); ++i) {
        thread_results[i] = performJoinThread(castSlices[i], titleSlices[i]);
    }

    // Für das Ergebnis soviel Speicher reservieren, wie die einzelnen Relationen groß sind
    size_t totalSize = 0;
    for (const auto& localResultRelation : thread_results) {
        totalSize += localResultRelation.size();
    }

    resultRelation.reserve(totalSize);


    // Ergebnisse zusammenführen
    for (const auto& vec : thread_results) {
        resultRelation.insert(resultRelation.end(), vec.begin(), vec.end());
    }

    return resultRelation;
}