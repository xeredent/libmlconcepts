cmake_minimum_required(VERSION 3.26.0)
project(mlconcepts VERSION 0.1.0 LANGUAGES C CXX)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

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
pybind11_add_module(mlconceptscore src/python_mlconcepts.cpp)
add_custom_target(python_mlconcepts ALL
                  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/mlconcepts
                  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/python/mlconcepts
                          ${CMAKE_CURRENT_BINARY_DIR}/mlconcepts
                  COMMAND ${CMAKE_COMMAND} -E rm -R 
                          ${CMAKE_CURRENT_BINARY_DIR}/mlconcepts/mlconceptscore
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                          $<TARGET_FILE:mlconceptscore>
                          ${CMAKE_CURRENT_BINARY_DIR}/mlconcepts/)
add_dependencies(python_mlconcepts mlconceptscore)

# Linting
add_custom_target(python_linting
                  COMMAND ruff check ${CMAKE_CURRENT_SOURCE_DIR}/src/python/mlconcepts)


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

# Install
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mlconcepts/ DESTINATION ${CMAKE_INSTALL_PREFIX})