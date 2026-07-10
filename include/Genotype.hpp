#pragma once

#include "Activation.hpp"
#include <cstddef>
#include <optional>
#include <vector>

class Rng;

struct NeuronGene {
  int neuron_id;
  double bias;
  Activation activation;
};

struct LinkId {
  int input_id;
  int output_id;
};

struct LinkGene {
  LinkId link_id;
  double weight;
  bool is_enabled;
};

class Genome {
private:
  Rng &rng_;
  int genome_id;
  int num_inputs;
  int num_outputs;
  std::vector<NeuronGene> neurons;
  std::vector<LinkGene> links;

public:
  Genome();
  Genome(Rng &rng, int id = -1, int inputs = 0, int outputs = 0);
  int get_genome_id() const;
  int get_num_inputs() const;
  int get_num_outputs() const;
  const std::vector<NeuronGene> &get_neurons() const;
  const std::vector<LinkGene> &get_links() const;
  void set_neuron_bias(int neuron_id, double new_bias);
  void set_link_weight(LinkId link_id, double new_weight);
  const std::optional<NeuronGene> find_neuron(const int neuron_id) const;
  const std::optional<LinkGene> find_link(const LinkId link_id_x) const;
  void add_neuron(NeuronGene newNeuron);
  void add_link(const LinkGene newLink);
  std::vector<int> get_input_ids() const;
  std::vector<int> get_output_ids() const;
  int get_num_neurons() const;
  std::vector<LinkGene>::iterator remove_link(const LinkId id_to_remove);
  void remove_hidden_neuron(const int neuron_id_to_remove);
  int get_next_neuron_id();
  bool isOutputNeuron(const int neuron_id);
  bool isInputNeuron(const int neuron_id);
  bool isHiddenNeuron(const int neuron_id);
};

const double kFitnessNotComputed = -1;

struct Individual {
  Genome genome;
  double fitness;
};
