#pragma once

#include "Activation.hpp"
#include "Genotype.hpp"
#include <vector>

struct NeuronInput {
  int input_id;
  double weight;
};

struct Neuron {
  double bias;
  std::vector<NeuronInput> inputs;
  Activation activation;
};

class FeedForwardNeuralNetwork {
private:
  std::vector<int> input_ids;
  std::vector<int> output_ids;
  std::vector<Neuron> neurons;
  static std::vector<std::vector<int>>
  toposort_neurons_into_layers(const std::vector<int> inputs,
                               const std::vector<int> outputs,
                               const std::vector<LinkGene> &links);

public:
  FeedForwardNeuralNetwork(std::vector<int> inputs, std::vector<int> outputs,
                           std::vector<Neuron> neurons);
  std::vector<double> outputs;
  static FeedForwardNeuralNetwork create_nn_from_gene(const Genome &genome);
};
