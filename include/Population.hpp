#pragma once

#include "Config.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"
#include <vector>
class Population {
private:
  std::vector<Individual> individuals_;
  Indexer genome_indexer_;
  Individual best_individual_;
  Rng &rng_;
  NeatConfig config_;

  Genome new_genome();
  void update_best();
  std::vector<Individual> reproduce();

public:
  Population(NeatConfig config, Rng &rng);

  template <typename FitnessFn> Individual run(FitnessFn compute_fitness) {
    for (int i = 0; i < config_.numGenerations; ++i) {
      compute_fitness(individuals_.begin(), individuals_.end());
      update_best();
      individuals_ = reproduce();
    }
  }
};
