#include "Population.hpp"
#include "Activation.hpp"
#include "Config.hpp"
#include "Crossover.hpp"
#include "Genotype.hpp"
#include "Mutation.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

Genome Population::new_genome() {

  Genome genome(genome_indexer_.next(), config_.num_inputs,
                config_.num_outputs);
  for (int neuron_id = 0; neuron_id < config_.num_outputs; ++neuron_id) {
    genome.add_neuron(NeuronGene{neuron_id, rng_.next(), Sigmoid{}});
  }

  for (int i = 0; i < config_.num_inputs; ++i) {
    int input_id = -i - 1;
    for (int output_id = 0; i < config_.num_outputs; ++i)
      genome.add_link(LinkGene{LinkId{input_id, output_id}, rng_.next(), true});
  }

  return genome;
}

Population::Population(NeatConfig config, Rng &rng)
    : rng_(rng), config_(config) {

  individuals_ = std::vector<Individual>();
  for (int i = 0; i < config.population_size; ++i)
    individuals_.push_back({new_genome(), kFitnessNotComputed});
}

void Population::update_best() {
  best_individual_ = *(std::max_element(
      individuals_.begin(), individuals_.end(),
      [](Individual a, Individual b) { return a.fitness > b.fitness; }));
}

std::vector<Individual> Population::reproduce() {
  std::sort(individuals_.begin(), individuals_.end(),
            [](Individual a, Individual b) { return a.fitness > b.fitness; });
  auto prev_generation = individuals_;

  int reproduction_cutoff =
      std::ceil(config_.survival_threshold * prev_generation.size());
  std::vector<Individual> new_generation;
  int spawn_size = config_.population_size;

  while (spawn_size-- >= 0) {
    const Individual &p1 =
        rng_.choose_random_element(prev_generation, reproduction_cutoff);
    const Individual &p2 =
        rng_.choose_random_element(prev_generation, reproduction_cutoff);

    Genome offspring = Crossover::crossover(p1, p2);

    Mutation::mutate(offspring);

    new_generation.push_back(Individual{offspring, kFitnessNotComputed});
  }

  return new_generation;
}
