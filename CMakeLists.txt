cmake_minimum_required(VERSION 3.5.0)
project(mlconcepts VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

include(CTest)
include_directories("include")
include_directories( ${EIGEN3_INCLUDE_DIRS} )


# Python module
pybind11_add_module(mlconcepts src/libmlconcepts.cpp)

# Test executables
add_executable(tests_partial_context tests/tests_partial_context.cpp)
add_executable(tests tests/tests.cpp)
target_link_libraries(tests_partial_context Eigen3::Eigen)
target_link_libraries(tests Eigen3::Eigen)

# CTest tests
add_test(NAME bitset64_serialization COMMAND tests_partial_context bitset64_serialization)
add_test(NAME bitset64_serialization_longsequences COMMAND tests_partial_context bitset64_serialization_longsequences)
add_test(NAME context64_serialization COMMAND tests_partial_context context64_serialization)
add_test(NAME small_context COMMAND tests_partial_context small_context)
add_test(NAME bitstreams1 COMMAND tests_partial_context bitstreams1)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)