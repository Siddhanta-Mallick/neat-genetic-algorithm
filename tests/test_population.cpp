#include "../external/catch.hpp"

#define private public
#include "Population.hpp"
#undef private

Rng rng_pop = Rng(1);

TEST_CASE("Population Initialization", "[Population]") {
  NeatConfig config;
  config.population_size = 15;

  Population pop(config, rng_pop);

  SECTION("Correct number of individuals are created") {
    REQUIRE(pop.individuals_.size() == 15);
  }

  SECTION("Initial fitness is set to kFitnessNotComputed") {
    for (const auto &ind : pop.individuals_) {
      REQUIRE(ind.fitness == kFitnessNotComputed);
    }
  }

  SECTION("Best individual is initialized with dummy genome") {
    REQUIRE(pop.best_individual_.genome.get_genome_id() == -1);
  }
}

TEST_CASE("Population update_best()", "[Population]") {
  NeatConfig config;
  config.population_size = 3;
  Population pop(config, rng_pop);

  // Manually assign fitnesses to test the max element logic
  pop.individuals_[0].fitness = 10.0;
  pop.individuals_[1].fitness = 50.0; // Expected Best
  pop.individuals_[2].fitness = 5.0;

  pop.update_best();

  SECTION("Best individual is updated correctly") {
    // NOTE: If you haven't fixed the std::max_element bug mentioned above,
    // this test will fail because your code will return 5.0 instead of 50.0.
    REQUIRE(pop.best_individual_.fitness == 50.0);
  }
}

TEST_CASE("Population reproduce()", "[Population]") {
  NeatConfig config;
  config.population_size = 10;
  config.survival_threshold = 0.5;
  Population pop(config, rng_pop);

  // Give them some arbitrary fitness scores so sorting logic executes properly
  for (int i = 0; i < config.population_size; ++i) {
    pop.individuals_[i].fitness = static_cast<double>(i);
  }

  std::vector<Individual> next_generation = pop.reproduce();

  SECTION("New generation size matches population size") {
    // NOTE: If you haven't fixed the while(spawn_size-- >= 0) bug,
    // this test will fail because next_generation.size() will be 11.
    REQUIRE(next_generation.size() == config.population_size);
  }

  SECTION("Offspring fitness is reset to kFitnessNotComputed") {
    for (const auto &ind : next_generation) {
      REQUIRE(ind.fitness == kFitnessNotComputed);
    }
  }
}
