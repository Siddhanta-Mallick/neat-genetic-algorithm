#include "Population.hpp"
#include "Config.hpp"
#include "Crossover.hpp"
#include "Genotype.hpp"
#include "Mutation.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

MutationConfig mutation_config{};

Population::Population(NeatConfig config, Rng &rng)
    : best_individual_(
          Individual(Genome(rng, -1, config.num_inputs, config.num_outputs))),
      rng_(rng), config_(config) {

  individuals_ = std::vector<Individual>();
  for (int i = 0; i < config.population_size; ++i)
    individuals_.push_back({Genome(rng, genome_indexer_.next(),
                                   config.num_inputs, config.num_outputs),
                            kFitnessNotComputed});
}

void Population::update_best() {
  best_individual_ = *(std::max_element(
      individuals_.begin(), individuals_.end(),
      [](Individual a, Individual b) { return a.fitness < b.fitness; }));
}

std::vector<Individual> Population::reproduce() {
  std::sort(individuals_.begin(), individuals_.end(),
            [](Individual a, Individual b) { return a.fitness > b.fitness; });
  auto prev_generation = individuals_;
  // TODO - do we need to preserve the top genomes for next generation ?
  int reproduction_cutoff =
      std::ceil(config_.survival_threshold * prev_generation.size());
  std::vector<Individual> new_generation;
  int spawn_size = config_.population_size;

  while (spawn_size-- > 0) {
    const Individual &p1 =
        rng_.choose_random_element(prev_generation, reproduction_cutoff);
    const Individual &p2 =
        rng_.choose_random_element(prev_generation, reproduction_cutoff);

    Genome offspring = Crossover::crossover(p1, p2);

    if (rng_.next_normal() < mutation_config.mutation_probability)
      Mutation::mutate(offspring);

    new_generation.push_back(Individual{offspring, kFitnessNotComputed});
  }

  return new_generation;
}
