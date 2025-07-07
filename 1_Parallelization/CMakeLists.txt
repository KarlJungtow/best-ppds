# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.21.0)
project(PPDS_PARALLELISM VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Optimization Flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O2 -march=native -mtune=native -fPIC")

set(PPDS_PROJECT_DIR "${CMAKE_SOURCE_DIR}/..")

include_directories(${PPDS_PROJECT_DIR}/Util/include)
if(NOT DEFINED DATA_DIRECTORY)
    set(DATA_DIRECTORY "${PPDS_PROJECT_DIR}/DataGenerators/data/")
endif()

if(NOT DEFINED SOURCE_DIRECTORY)
    set(SOURCE_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endif()

# Load gtest
include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG        v1.14.0
    )

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    set(BUILD_SHARED_LIBS TRUE)  # Forces shared libraries, which implies -fPIC
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()
FetchContent_MakeAvailable(googletest)

# Define the shared library
add_library(1_Parallelization SHARED Join.cpp)

# Define the executable target that uses the shared library
add_executable(ParallelismExecutable Join.cpp)

# Link with Libraries
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(1_Parallelization PUBLIC OpenMP::OpenMP_CXX gtest_main)
    target_link_libraries(ParallelismExecutable OpenMP::OpenMP_CXX gtest_main)
else (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found")
    message(STATUS "On Ubuntu, install the package by the following command!")
    message(STATUS "sudo apt install libomp-dev")
    target_link_libraries(1_Parallelization PUBLIC gtest_main)
    target_link_libraries(ParallelismExecutable gtest_main)
endif (OpenMP_CXX_FOUND)

# If necessary, include gtest include directories
target_include_directories(1_Parallelization PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_BINARY_DIR})
target_include_directories(ParallelismExecutable PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_BINARY_DIR})

# Add compile definitions
target_compile_definitions(1_Parallelization PRIVATE
    DATA_DIRECTORY="${DATA_DIRECTORY}"
    SOURCE_DIRECTORY="${SOURCE_DIRECTORY}"
    )

target_compile_definitions(ParallelismExecutable PRIVATE
    DATA_DIRECTORY="${DATA_DIRECTORY}"
    SOURCE_DIRECTORY="${SOURCE_DIRECTORY}"
    )

