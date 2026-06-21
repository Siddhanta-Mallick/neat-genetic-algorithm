#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <vector>

#include "Activation.hpp"

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
  int genome_id;
  int num_inputs;
  int num_outputs;
  std::vector<NeuronGene> neurons;
  std::vector<LinkGene> links;

public:
  Genome(int id = -1, int inputs = 0, int outputs = 0)
      : genome_id(id), num_inputs(inputs), num_outputs(outputs) {}
  int get_genome_id() const { return genome_id; }
  int get_num_inputs() const { return num_inputs; }
  int get_num_ouputs() const { return num_outputs; }
  const std::vector<NeuronGene> &get_neurons() const { return neurons; }
  const std::vector<LinkGene> &get_links() const { return links; }
  const std::optional<NeuronGene> find_neuron(const int neuron_id) const {
    auto it = std::find_if(neurons.begin(), neurons.end(),
                           [neuron_id](const NeuronGene &neuron) {
                             return neuron.neuron_id == neuron_id;
                           });
    return it != neurons.end() ? std::optional<NeuronGene>(*it) : std::nullopt;
  }
  const std::optional<LinkGene> find_link(const LinkId link_id_x) const {
    auto it = std::find_if(
        links.begin(), links.end(), [link_id_x](const LinkGene &link) {
          return link_id_x.input_id == link.link_id.input_id &&
                 link_id_x.output_id == link.link_id.output_id;
        });

    return it != links.end() ? std::optional<LinkGene>(*it) : std::nullopt;
  }
  void add_neuron(const NeuronGene newNeuron) { neurons.push_back(newNeuron); }
  void add_link(const LinkGene newLink) { links.push_back(newLink); }

  std::vector<int> get_input_ids() const {
    std::vector<int> input_ids;
    for (int i = 0; i < num_inputs; ++i)
      input_ids.push_back(-i - 1);
    return input_ids;
  }

  std::vector<int> get_output_ids() const {
    std::vector<int> output_ids;
    for (int i = 0; i < num_outputs; ++i)
      output_ids.push_back(i);
    return output_ids;
  }

  int get_num_neurons() const { return neurons.size(); }

  void remove_link(LinkGene &link_to_remove) {
    if (links.empty())
      return;

    std::ptrdiff_t index = &link_to_remove - &links[0];
    if (index >= 0 && static_cast<size_t>(index) < links.size())
      links.erase(links.begin() + index);
  }
};

const double kFitnessNotComputed = -1;

struct Individual {
  Genome genome;
  double fitness;
};
