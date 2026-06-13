#pragma once

#include "Activation.hpp"
#include "Genotype.hpp"
#include <vector>

struct NeuronInput
{
  int input_id;
  double weight;
};

struct Neuron
{
  double bias;
  std::vector<NeuronInput> inputs;
  ActivationFn activation;
};

class FeedForwardNeuralNetwork
{
private:
  std::vector<int> input_ids;
  std::vector<int> output_ids;
  std::vector<Neuron> neurons;

public:
  std::vector<double> outputs;
  static FeedForwardNeuralNetwork create_nn_from_gene(const Genome &genome);
};
