#include "TimerUtil.hpp"
#include "JoinUtils.hpp"
#include "absl/container/flat_hash_map.h"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <gtest/gtest.h>
#include <omp.h>
#include <thread>


std::vector<ResultRelation> performJoin(const std::vector<CastRelation>& castRelation,
                                        const std::vector<TitleRelation>& titleRelation,
                                        int numThreads) {

    // Using Google's Hashmap go brr
    absl::flat_hash_map<int, TitleRelation> titleMap;

    //Speicherplatz reservieren
    titleMap.reserve(titleRelation.size());

    // Hashhmap füllen
    for (const auto& title : titleRelation) {
        titleMap.emplace(title.titleId, title);
    }

    //Jeder Thread bekommt einen Vektor
    std::vector<std::vector<ResultRelation>> threadLocalResults(numThreads);
    omp_set_num_threads(numThreads);

#pragma omp parallel
    {

        int threadId = omp_get_thread_num();
        //Greife auf den richtigen Threadspeicher zu
        std::vector<ResultRelation>& localResult = threadLocalResults[threadId];
        //Reserviere genug (ist es genug?) Platz für gejointe tupel
        localResult.reserve((castRelation.size() / numThreads) * 1.25);

        // Round-Robin approach for cast-Relation casts
#pragma omp for schedule(static, 508) nowait
        for (const auto& cast : castRelation) {
            auto it = titleMap.find(cast.movieId);
            if (it != titleMap.end()) {
                localResult.push_back(createResultTuple(cast, it->second));
            }
        }
    }

    // Merge thread-local results
    std::vector<ResultRelation> resultTuples;
    // Estimate size, optionally after profiling average match rate
    resultTuples.reserve(castRelation.size());

    for (auto& local : threadLocalResults) {
        resultTuples.insert(resultTuples.end(),
                            std::make_move_iterator(local.begin()),
                            std::make_move_iterator(local.end()));
    }

    return resultTuples;
}


TEST(ParallelizationTest, TestJoiningTuples) {
    std::cout << "Test reading data from a file.\n";
    const auto leftRelation = loadCastRelation(DATA_DIRECTORY + std::string("cast_info_uniform.csv"), 1000000);
    const auto rightRelation = loadTitleRelation(DATA_DIRECTORY + std::string("title_info_uniform.csv"), 1000000);

    Timer timer("Parallelized Join execute");
    timer.start();

    auto resultTuples = performJoin(leftRelation, rightRelation, 8);

    timer.pause();

    std::cout << "Timer: " << timer << std::endl;
    std::cout << "Result size: " << resultTuples.size() << std::endl;
    std::cout << "\n\n";
}