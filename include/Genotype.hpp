#pragma once

#include <vector>

#include "Activation.hpp"

struct NeuronGene
{
    int neuron_id;
    double bias;
    Activation activation;
};

struct LinkId
{
    int input_id;
    int output_id;
};

struct LinkGene
{
    LinkId link_id;
    double weight;
    bool is_enabled;
};

struct Genome
{
    int genome_id;
    int num_inputs;
    int num_outputs;
    std::vector<NeuronGene> neurons;
    std::vector<LinkGene> links;
};