#include <assert.h>

#include "include/Genotype.hpp"
#include "include/Activation.hpp"
#include "include/Utils.hpp"

Rng rng(std::nullopt);

NeuronGene crossover_neuron(const NeuronGene &a, NeuronGene &b)
{
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(a.bias, b.bias);
    Activation activation = rng.choose(a.activation, b.activation);
    return {neuron_id, bias, activation};
}