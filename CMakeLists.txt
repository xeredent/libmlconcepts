cmake_minimum_required(VERSION 3.26.0)
project(mlconcepts VERSION 0.1.0 LANGUAGES CXX)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Set flags. User flags are first checked against the target architecture compiler
if(DEFINED ENV{LIBMLCONCEPTS_ISET_DISTRIBUTION})
    # Custom flags are checked before applying
    include(CheckCXXCompilerFlag)
    unset(COMPILER_SUPPORTS_ISET_DISTR CACHE)
    CHECK_CXX_COMPILER_FLAG("$ENV{LIBMLCONCEPTS_ISET_DISTRIBUTION}" COMPILER_SUPPORTS_ISET_DISTR)
    if (COMPILER_SUPPORTS_ISET_DISTR)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 $ENV{LIBMLCONCEPTS_ISET_DISTRIBUTION}")
    else()
        message(WARNING "Flags '$ENV{LIBMLCONCEPTS_ISET_DISTRIBUTION}' are not supported by the compiler for this target. Proceeding with just -O3.")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

# Find Dependencies
find_package (Eigen3 3.4 NO_MODULE)
if (NOT TARGET Eigen3::Eigen) #If eigen is not found, download it
        if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0)
                if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/eigen3.4.0.zip)
                        file(DOWNLOAD 
                             https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip 
                             ${CMAKE_CURRENT_SOURCE_DIR}/eigen3.4.0.zip)
                endif()
                execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf 
                                ${CMAKE_CURRENT_SOURCE_DIR}/eigen3.4.0.zip
                                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
        include_directories("eigen-3.4.0")
else()
    include_directories( ${EIGEN3_INCLUDE_DIRS} )
endif ()

#Find Python/Pybind
if(WIN32)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11-2.12.0)
        if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11-2.12.0zip)
                file(DOWNLOAD 
                    https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
                    ${CMAKE_CURRENT_SOURCE_DIR}/pybind11-2.12.0.zip)
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf 
                        ${CMAKE_CURRENT_SOURCE_DIR}/pybind11-2.12.0.zip
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    add_subdirectory(pybind11-2.12.0)
else()
    if (DEFINED Python_EXECUTABLE)
        set(PYBIND11_FINDPYTHON ON)
    else()
        find_package(Python COMPONENTS Interpreter Development.Module)
    endif()
    find_package(pybind11 CONFIG REQUIRED)
endif()

include(CTest)
include_directories("include")

# Python module
pybind11_add_module(mlconceptscore src/python_mlconcepts.cpp)
target_link_libraries(mlconceptscore PRIVATE pybind11::module)
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