#include "JoinUtils.hpp"
#include <vector>
#include <memory>
#include <cctype>
#include <string_view>
#include <omp.h>
#include <algorithm>
using namespace std;

// Konstanten für Trie-Konfiguration
static constexpr int ALPHABET_SIZE = 36;
static constexpr int OTHER_INDEX = ALPHABET_SIZE;
static constexpr int TOTAL_CHILDREN = ALPHABET_SIZE + 1;

class Trie {
private:
    struct TrieNode {
        bool endOfWord = false;
        vector<const CastRelation*> cast;
        unique_ptr<TrieNode> children[TOTAL_CHILDREN];
    };

    unique_ptr<TrieNode> root;

    static int charToIndex(char c) {
        c = tolower(c);
        if (c >= 'a' && c <= 'z') {
            return c - 'a'; // 0 to 25
        } else if (c >= '0' && c <= '9') {
            return 26 + (c - '0'); // 26 to 35
        }
        return OTHER_INDEX; // anything else
    }

    static void mergeNodes(TrieNode* this_node, TrieNode* other_node) {
        if (other_node->endOfWord) {
            this_node->endOfWord = true;
            for (const auto* castPtr : other_node->cast) {
                this_node->cast.push_back(castPtr);
            }
        }

        for (int i = 0; i < TOTAL_CHILDREN; i++) {
            if (other_node->children[i]) {
                if (!this_node->children[i]) {
                    this_node->children[i] = move(other_node->children[i]);
                } else {
                    mergeNodes(this_node->children[i].get(), other_node->children[i].get());
                }
            }
        }
    }

public:
    Trie() : root(make_unique<TrieNode>()) {}

    void insert(const CastRelation* cast) const {
        TrieNode* node = root.get();
        string_view note(cast->note);

        for (char c : note) {
            int index = charToIndex(c);
            if (!node->children[index]) {
                node->children[index] = make_unique<TrieNode>();
            }
            node = node->children[index].get();
        }

        node->endOfWord = true;
        node->cast.push_back(cast);
    }

    void findPrefixMatches(const string& prefix, vector<const CastRelation*>& results) const {
        TrieNode* node = root.get();
        for (char c : prefix) {
            if (node->endOfWord) {
                results.insert(results.end(), node->cast.begin(), node->cast.end());
            }
            int index = charToIndex(c);
            if (!node->children[index]) return;
            node = node->children[index].get();
        }
        if (node->endOfWord) {
            results.insert(results.end(), node->cast.begin(), node->cast.end());
        }
    }

    void merge(Trie&& other) {
        mergeNodes(root.get(), other.root.get());
    }
};

//-------------------------------------------------------------------------------------------------------------------------

vector<ResultRelation> performJoin(const vector<CastRelation>& castRelation,
                                    const vector<TitleRelation>& titleRelation,
                                    int numThreads) {
    // Setze die Anzahl der OpenMP-Threads
    omp_set_num_threads(numThreads);

    vector<Trie> localTries(numThreads);

    // Phase 1: Paralleles Einfügen in lokale Tries
    #pragma omp parallel num_threads(numThreads)
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (const auto & i : castRelation) {
            localTries[thread_id].insert(&i);
        }
    }

    // Phase 2: Zusammenführen der lokalen Tries
    Trie globalTrie;
    for (auto& lt : localTries) {
        globalTrie.merge(move(lt));
    }

    // Phase 3: Paralleles Suchen und Sammeln der Ergebnisse
    vector<ResultRelation> globalResults;
    vector<vector<ResultRelation>> threadResults(numThreads);

    #pragma omp parallel num_threads(numThreads)
    {
        int thread_id = omp_get_thread_num();
        vector<ResultRelation>& localResults = threadResults[thread_id];
        localResults.reserve(titleRelation.size() * 2 / numThreads); // Heuristische Reserve

        #pragma omp for schedule(dynamic, 256)
        for (const auto & title : titleRelation) {
            vector<const CastRelation*> prefixMatches;
            prefixMatches.reserve(10);
            globalTrie.findPrefixMatches(title.title, prefixMatches);
            for (const auto* cast : prefixMatches) {
                localResults.emplace_back(createResultTuple(*cast, title));
            }
        }
    }

    // Kombiniere alle lokalen Ergebnisse
    size_t totalSize = 0;
    for (auto& vec : threadResults) {
        totalSize += vec.size();
    }
    globalResults.reserve(totalSize);
    for (auto& vec : threadResults) {
        move(vec.begin(), vec.end(), back_inserter(globalResults));
    }

    return globalResults;
}