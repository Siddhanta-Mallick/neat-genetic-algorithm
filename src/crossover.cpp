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

LinkGene crossover_link(const LinkGene &a, LinkGene &b)
{
    assert(a.link_id.input_id == b.link_id.input_id && a.link_id.output_id == b.link_id.output_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(a.weight, b.weight);
    bool is_enabled = rng.choose(a.is_enabled, b.is_enabled);
    return {link_id, weight, is_enabled};
}