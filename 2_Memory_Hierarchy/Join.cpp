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

//----------------------------------------------------------------------------------------------------------------------------
CastRelation makeCast(int id, int pid, int mid, int prid, const std::string& note, int order, int rid) {
    CastRelation c{};
    c.castInfoId = id;
    c.personId = pid;
    c.movieId = mid;
    c.personRoleId = prid;
    strncpy(c.note, note.c_str(), sizeof(c.note));
    c.note[sizeof(c.note) - 1] = '\0'; // Ensure null-terminated
    c.nrOrder = order;
    c.roleId = rid;
    return c;
}

TitleRelation makeTitle(int tid, const std::string& title, const std::string& index, int kind, int year, int imdb, const std::string& code) {
    TitleRelation t{};
    t.titleId = tid;
    strncpy(t.title, title.c_str(), sizeof(t.title));
    t.title[sizeof(t.title) - 1] = '\0';

    strncpy(t.imdbIndex, index.c_str(), sizeof(t.imdbIndex));
    t.imdbIndex[sizeof(t.imdbIndex) - 1] = '\0';

    t.kindId = kind;
    t.productionYear = year;
    t.imdbId = imdb;

    strncpy(t.phoneticCode, code.c_str(), sizeof(t.phoneticCode));
    t.phoneticCode[sizeof(t.phoneticCode) - 1] = '\0';

    // Leave other fields zero for simplicity
    return t;
}

int main() {std::vector<CastRelation> castRelations = {
    makeCast(1, 101, 10, 1, "note1", 0, 1),
    makeCast(2, 102, 10, 2, "note2", 1, 2),
    makeCast(3, 103, 11, 1, "note3", 2, 1),
    makeCast(4, 104, 12, 1, "note4", 3, 1),
    makeCast(5, 105, 12, 2, "note5", 4, 3),
    makeCast(6, 106, 13, 1, "note6", 5, 2),
    makeCast(7, 107, 14, 1, "note7", 6, 1)
};

    std::vector<TitleRelation> titleRelations = {
        makeTitle(10, "Title A", "", 1, 2001, 1001, ""),
        makeTitle(10, "Title A.1", "", 1, 2001, 1002, ""),
        makeTitle(11, "Title B", "", 1, 2002, 1003, ""),
        makeTitle(12, "Title C", "", 1, 2003, 1004, ""),
        makeTitle(13, "Title D", "", 1, 2004, 1005, ""),
        makeTitle(14, "Title E", "", 1, 2005, 1006, "")
    };


    // Pick cutoff indices somewhere in the middle
    int cast_cutoff_index = 5;  // Points to movieId = 13
    int title_cutoff_index = 4; // Points to titleId = 13

    std::cout << "=== Original CastRelation Entries ===" << std::endl;
    for (const auto& c : castRelations) {
        std::cout << castRelationToString(c) << std::endl;
    }

    std::cout << "\n=== Original TitleRelation Entries ===" << std::endl;
    for (const auto& t : titleRelations) {
        std::cout << titleRelationToString(t) << std::endl;
    }

    // Perform splitting
    int split_cast = splitCast(castRelations, cast_cutoff_index);
    int split_title = splitTitle(titleRelations, title_cutoff_index);

    std::cout << "\n--- splitCast at index " << cast_cutoff_index << " returned: " << split_cast << std::endl;
    std::cout << "--- splitTitle at index " << title_cutoff_index << " returned: " << split_title << std::endl;

    // Use splitRelations to determine safe cutoff for parallel chunks
    std::vector<int> splits = splitRelations(castRelations, titleRelations, cast_cutoff_index, title_cutoff_index);

    std::cout << "\n=== splitRelations Output ===" << std::endl;
    std::cout << "Title split index: " << splits[0] << std::endl;
    std::cout << "Cast split index:  " << splits[1] << std::endl;

    std::cout << "\n=== Cast Slice ===" << std::endl;
    for (int i = 0; i < splits[1]; ++i) {
        std::cout << castRelationToString(castRelations[i]) << std::endl;
    }

    std::cout << "\n=== Title Slice ===" << std::endl;
    for (int i = 0; i < splits[0]; ++i) {
        std::cout << titleRelationToString(titleRelations[i]) << std::endl;
    }

    return 0;
}