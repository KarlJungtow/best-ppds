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

#ifndef JOINUTIL_HPP
#define JOINUTIL_HPP

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

//==--------------------------------------------------------------------==//
//==------------------ RELATION & RELATION UTILITY----------------------==//
//==--------------------------------------------------------------------==//

static constexpr size_t NUM_FIELD_CAST_RELATION = 7;
    static constexpr size_t NUM_FIELDS_TITLE_RELATION = 12;

    struct CastRelation {
      int32_t castInfoId;
      int32_t personId;
      int32_t movieId;
      int32_t personRoleId;
      char note[100];
      int32_t nrOrder;
      int32_t roleId;
    };

    // Production year would be a good to add (extract from title string).
    struct TitleRelation {
      int32_t titleId;
      char title[200];
      char imdbIndex[12];
      int32_t kindId;
      int32_t productionYear;
      int32_t imdbId;
      char phoneticCode[5];
      int32_t episodeOfId;
      int32_t seasonNr;
      int32_t episodeNr;
      char seriesYears[49];
      char md5sum[32];
    };

    struct ResultRelation {
      int32_t titleId;
      char title[200];
      char imdbIndex[12];
      int32_t kindId;
      int32_t productionYear;
      int32_t imdbId;
      char phoneticCode[5];
      int32_t episodeOfId;
      int32_t seasonNr;
      int32_t episodeNr;
      char seriesYears[49];
      char md5sum[32];
      int32_t castInfoId;
      int32_t personId;
      int32_t movieId;
      int32_t personRoleId;
      char note[100];
      int32_t nrOrder;
      int32_t roleId;
    };

    inline bool operator==(const ResultRelation& lhs, const ResultRelation& rhs) {
      return lhs.titleId == rhs.titleId &&
             (std::memcmp(lhs.title, rhs.title, sizeof(lhs.title)) == 0) &&
             (std::memcmp(lhs.imdbIndex, rhs.imdbIndex, sizeof(lhs.imdbIndex)) == 0) &&
             lhs.kindId == rhs.kindId &&
             lhs.productionYear == rhs.productionYear &&
             lhs.imdbId == rhs.imdbId &&
             (std::memcmp(lhs.phoneticCode, rhs.phoneticCode, sizeof(lhs.phoneticCode)) == 0) &&
             lhs.episodeOfId == rhs.episodeOfId &&
             lhs.seasonNr == rhs.seasonNr &&
             lhs.episodeNr == rhs.episodeNr &&
             (std::memcmp(lhs.seriesYears, rhs.seriesYears, sizeof(lhs.seriesYears)) == 0) &&
             (std::memcmp(lhs.md5sum, rhs.md5sum, sizeof(lhs.md5sum)) == 0) &&
             lhs.castInfoId == rhs.castInfoId &&
             lhs.personId == rhs.personId &&
             lhs.personRoleId == rhs.personRoleId &&
             (std::memcmp(lhs.note, rhs.note, sizeof(lhs.note)) == 0) &&
             lhs.nrOrder == rhs.nrOrder &&
             lhs.roleId == rhs.roleId;
    }

    inline bool operator<(const ResultRelation& lhs, const ResultRelation& rhs) {
      if (lhs.titleId != rhs.titleId) return lhs.titleId < rhs.titleId;
      if (std::strcmp(lhs.title, rhs.title) != 0) return std::strcmp(lhs.title, rhs.title) < 0;
      if (std::strcmp(lhs.imdbIndex, rhs.imdbIndex) != 0) return std::strcmp(lhs.imdbIndex, rhs.imdbIndex) < 0;
      if (lhs.kindId != rhs.kindId) return lhs.kindId < rhs.kindId;
      if (lhs.productionYear != rhs.productionYear) return lhs.productionYear < rhs.productionYear;
      if (lhs.imdbId != rhs.imdbId) return lhs.imdbId < rhs.imdbId;
      if (std::strcmp(lhs.phoneticCode, rhs.phoneticCode) != 0) return std::strcmp(lhs.phoneticCode, rhs.phoneticCode) < 0;
      if (lhs.episodeOfId != rhs.episodeOfId) return lhs.episodeOfId < rhs.episodeOfId;
      if (lhs.seasonNr != rhs.seasonNr) return lhs.seasonNr < rhs.seasonNr;
      if (lhs.episodeNr != rhs.episodeNr) return lhs.episodeNr < rhs.episodeNr;
      if (std::strcmp(lhs.seriesYears, rhs.seriesYears) != 0) return std::strcmp(lhs.seriesYears, rhs.seriesYears) < 0;
      if (std::strcmp(lhs.md5sum, rhs.md5sum) != 0) return std::strcmp(lhs.md5sum, rhs.md5sum) < 0;
      if (lhs.castInfoId != rhs.castInfoId) return lhs.castInfoId < rhs.castInfoId;
      if (lhs.personId != rhs.personId) return lhs.personId < rhs.personId;
      if (lhs.movieId != rhs.movieId) return lhs.movieId < rhs.movieId;
      if (lhs.personRoleId != rhs.personRoleId) return lhs.personRoleId < rhs.personRoleId;
      if (std::strcmp(lhs.note, rhs.note) != 0) return std::strcmp(lhs.note, rhs.note) < 0;
      if (lhs.nrOrder != rhs.nrOrder) return lhs.nrOrder < rhs.nrOrder;
      return lhs.roleId < rhs.roleId; // Last comparison to fully define the ordering
    }

    [[nodiscard]] inline std::string titleRelationToString(const TitleRelation& relation) {
      std::ostringstream oss;
      oss<< relation.titleId << ","
          << relation.title << ","
          << relation.imdbIndex << ","
          << relation.kindId << ","
          << relation.productionYear << ","
          << relation.imdbId << ","
          << relation.phoneticCode << ","
          << relation.episodeOfId << ","
          << relation.seasonNr << ","
          << relation.episodeNr << ","
          << relation.seriesYears << ","
          << relation.md5sum;

      return oss.str();
    }

    [[nodiscard]] inline std::string castRelationToString(const CastRelation& relation) {
      std::ostringstream oss;
      oss << relation.castInfoId << ","
          << relation.personId << ","
          << relation.movieId << ","
          << relation.personRoleId << ","
          << relation.note << ","
          << relation.nrOrder << ","
          << relation.roleId;

      return oss.str();
    }

    [[nodiscard]] inline std::string resultRelationToString(const ResultRelation& relation) {
      std::ostringstream oss;
      oss << relation.titleId << ","
          << relation.title << ","
          << relation.imdbIndex << ","
          << relation.kindId << ","
          << relation.productionYear << ","
          << relation.imdbId << ","
          << relation.phoneticCode << ","
          << relation.episodeOfId << ","
          << relation.seasonNr << ","
          << relation.episodeNr << ","
          << relation.seriesYears << ","
          << relation.md5sum << ","
          << relation.castInfoId << ","
          << relation.personId << ","
          << relation.movieId << ","
          << relation.personRoleId << ","
          << relation.note << ","
          << relation.nrOrder << ","
          << relation.roleId;

      return oss.str();
    }

    inline void assignValueFromString(TitleRelation& titleRelation, const std::string& value, const size_t fieldIndex) {
      switch (fieldIndex) {
      case 0: titleRelation.titleId = std::stoi(value); break;
      case 1: std::memcpy(titleRelation.title, value.c_str(), 200); break;
      case 2: std::memcpy(titleRelation.imdbIndex, value.c_str(), 12); break;
      case 3: titleRelation.kindId = std::stoi(value); break;
      case 4: titleRelation.productionYear = std::stoi(value); break;
      case 5: titleRelation.imdbId = std::stoi(value); break;
      case 6: std::memcpy(titleRelation.phoneticCode, value.c_str(), 5); break;
      case 7: titleRelation.episodeOfId = std::stoi(value); break;
      case 8: titleRelation.seasonNr = std::stoi(value); break;
      case 9: titleRelation.episodeNr = std::stoi(value); break;
      case 10: std::memcpy(titleRelation.seriesYears, value.c_str(), 49); break;
      case 11: std::memcpy(titleRelation.md5sum, value.c_str(), 32); break;
      default: break;
      }
    }

    inline void assignValueFromString(CastRelation& castRelation, const std::string& value, const size_t fieldIndex) {
      switch (fieldIndex) {
      case 0: castRelation.castInfoId = std::stoi(value); break;
      case 1: castRelation.personId = std::stoi(value); break;
      case 2: castRelation.movieId = std::stoi(value); break;
      case 3: castRelation.personRoleId = std::stoi(value); break;
      case 4: std::memcpy(castRelation.note, value.c_str(), 100); break;
      case 5: castRelation.nrOrder = std::stoi(value); break;
      case 6: castRelation.roleId = std::stoi(value); break;
      default: break;
      }
    }

    //==--------------------------------------------------------------------==//
    //==--------------------- DATASET LOADING LOGIC ------------------------==//
    //==--------------------------------------------------------------------==//

    inline bool parseLine(const std::string& line, TitleRelation& record) {
      std::istringstream ss(line);
      std::string field;
      size_t fieldIndex = 0;

      while (std::getline(ss, field, ',')) {
        if (fieldIndex >= NUM_FIELDS_TITLE_RELATION) {
          std::cerr << "Error: Too many fields in CSV line" << std::endl;
          return false;
        }
        assignValueFromString(record, field, fieldIndex);
        fieldIndex++;
      }

      if (fieldIndex != NUM_FIELDS_TITLE_RELATION) {
        std::cerr << "Error: Too few fields in CSV line" << std::endl;
        return false;
      }

      return true;
    }

    inline bool parseLine(const std::string& line, CastRelation& record) {
      std::istringstream ss(line);
      std::string field;
      size_t fieldIndex = 0;

      while (std::getline(ss, field, ',')) {
        if (fieldIndex >= NUM_FIELD_CAST_RELATION) {
          std::cerr << "Error: Too many fields in CSV line" << std::endl;
          return false;
        }
        assignValueFromString(record, field, fieldIndex);
        fieldIndex++;
      }

      if (fieldIndex != NUM_FIELD_CAST_RELATION) {
        std::cerr << "Error: Too few fields in CSV line" << std::endl;
        return false;
      }

      return true;
    }

    template <typename Relation>
    std::vector<Relation> load(const std::string& filename, const size_t numberOfTuples = SIZE_MAX) {
      std::vector<Relation> data;
      std::ifstream file(filename);
      if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        exit(-1);
      }

      std::string line;
      bool firstLine = true;
      while (std::getline(file, line)) {
        if (firstLine) {
          firstLine = false;
          continue;
        }
        Relation record;
        if (parseLine(line, record)) {
          data.emplace_back(record);
        } else {
          std::cerr << "Error: Failed to parse line: " << line << std::endl;
        }

        if (data.size() >= numberOfTuples) {
          std::cout << "Loaded enough tuples. Returning now..." << std::endl;
          break;
        }
      }

      file.close();
      std::cout << "Loaded " << data.size() << " tuples from file." << std::endl;
      return data;
    }

    inline std::vector<TitleRelation> loadTitleRelation(const std::string& filename, const size_t numberOfTuples = SIZE_MAX) {
      return load<TitleRelation>(filename, numberOfTuples);
    }

    inline std::vector<CastRelation> loadCastRelation(const std::string& filename, const size_t numberOfTuples = SIZE_MAX) {
      return load<CastRelation>(filename, numberOfTuples);
    }

    inline ResultRelation createResultTuple(const CastRelation& cast, const TitleRelation& title) {
      ResultRelation result;
      // Assign values from title to result
      result.titleId = title.titleId;
      std::memcpy(result.title, title.title, 200);
      std::memcpy(result.imdbIndex, title.imdbIndex, 12);
      result.kindId = title.kindId;
      result.productionYear = title.productionYear;
      result.imdbId = title.imdbId;
      std::memcpy(result.phoneticCode, title.phoneticCode, 5);
      result.episodeOfId = title.episodeOfId;
      result.seasonNr = title.seasonNr;
      result.episodeNr = title.episodeNr;
      std::memcpy(result.seriesYears, title.seriesYears, 49);
      std::memcpy(result.md5sum, title.md5sum, 32);

      // Assign values from castInfo to result
      result.castInfoId = cast.castInfoId;
      result.personId = cast.personId;
      result.movieId = cast.movieId;
      result.personRoleId = cast.personRoleId;
      std::memcpy(result.note, cast.note, 100);
      result.nrOrder = cast.nrOrder;
      result.roleId = cast.roleId;

      return result;
    }

#endif //JOINUTIL_HPP
