#pragma once

#include "Genotype.hpp"
#include "Utils.hpp"

class Crossover {
private:
  static NeuronGene crossover_neuron(const NeuronGene &a, NeuronGene &b);
  static LinkGene crossover_link(const LinkGene &a, LinkGene &b);
  static Genome crossover_individual(const Individual &dominant,
                                     const Individual &recessive);

public:
  static Rng &rng_;
  static Indexer &genome_indexer_;
  static Genome crossover(const Individual &p1, const Individual &p2);
};
