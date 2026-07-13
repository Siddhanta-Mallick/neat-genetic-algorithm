cmake_minimum_required(VERSION 3.23)
project(NEAT_Library VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# Core Library 
add_library(neat_core STATIC
  src/Crossover.cpp
  src/FeedForwardNeuralNetwork.cpp
  src/Genotype.cpp
  src/Mutation.cpp
  src/Population.cpp
)

target_include_directories(neat_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Executable tests
add_executable(neat_tests
    tests/test_main.cpp
    tests/test_cycle.cpp
    tests/test_mutation.cpp
    tests/test_genome.cpp
    tests/test_crossover.cpp
    tests/test_feedforwardneuralnetwork.cpp
    tests/test_population.cpp
)

target_link_libraries(neat_tests PRIVATE neat_core)
target_include_directories(neat_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
add_test(NAME RunNeatTests COMMAND neat_tests)

# Uncomment below for address sanitizer, causes slow execution
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
