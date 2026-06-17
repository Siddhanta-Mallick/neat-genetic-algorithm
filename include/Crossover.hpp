#pragma once

#include "Utils.hpp"
#include "Genotype.hpp"

class Crossover
{
private:
    Indexer &genome_indexer_;
    Rng &rng_;
    NeuronGene crossover_neuron(const NeuronGene &a, NeuronGene &b);
    LinkGene crossover_link(const LinkGene &a, LinkGene &b);
    Genome crossover_individual(const Individual &dominant, const Individual &recessive);

public:
    Crossover(Indexer &genome_indexer, Rng &rng);
    Genome crossover(const Individual &p1, const Individual &p2);
};
