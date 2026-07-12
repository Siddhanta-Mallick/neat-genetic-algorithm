#include "Genotype.hpp"
#include "Utils.hpp"
#include <algorithm>

Genome::Genome(Rng &rng, int id, int inputs, int outputs)
    : rng_(rng), genome_id(id), num_inputs(inputs), num_outputs(outputs) {
  for (int i = 0; i < num_inputs; i++)
    neurons.push_back(NeuronGene{-i - 1, rng_.get().next_normal(), ReLU{}});
  for (int i = 0; i < num_outputs; i++)
    neurons.push_back(NeuronGene{i, rng_.get().next_normal(), Sigmoid{}});

  for (int i = 0; i < num_inputs; ++i) {
    int input_id = -i - 1;
    for (int output_id = 0; output_id < num_outputs; ++output_id) {
      links.push_back(LinkGene{LinkId{input_id, output_id},
                               rng_.get().next_normal(), true});
    }
  }
}

Genome::Genome(Rng &rng, int id) : rng_(rng), genome_id(id) {}

int Genome::get_genome_id() const { return genome_id; }

int Genome::get_num_inputs() const { return num_inputs; }

int Genome::get_num_outputs() const { return num_outputs; }

const std::vector<NeuronGene> &Genome::get_neurons() const { return neurons; }

const std::vector<LinkGene> &Genome::get_links() const { return links; }

const std::optional<NeuronGene> Genome::find_neuron(const int neuron_id) const {
  auto it = std::find_if(neurons.begin(), neurons.end(),
                         [neuron_id](const NeuronGene &neuron) {
                           return neuron.neuron_id == neuron_id;
                         });
  return it != neurons.end() ? std::optional<NeuronGene>(*it) : std::nullopt;
}

const std::optional<LinkGene> Genome::find_link(const LinkId link_id_x) const {
  auto it = std::find_if(links.begin(), links.end(),
                         [link_id_x](const LinkGene &link) {
                           return link_id_x.input_id == link.link_id.input_id &&
                                  link_id_x.output_id == link.link_id.output_id;
                         });

  return it != links.end() ? std::optional<LinkGene>(*it) : std::nullopt;
}

void Genome::add_neuron(NeuronGene newNeuron) { neurons.push_back(newNeuron); }

void Genome::add_link(const LinkGene newLink) {
  bool isValidInput = find_neuron(newLink.link_id.input_id).has_value();
  bool isValidOutput = find_neuron(newLink.link_id.output_id).has_value();
  if (isValidInput && isValidOutput)
    links.push_back(newLink);
}

std::vector<int> Genome::get_input_ids() const {
  std::vector<int> input_ids;
  for (int i = 0; i < num_inputs; ++i)
    input_ids.push_back(-i - 1);
  return input_ids;
}

std::vector<int> Genome::get_output_ids() const {
  std::vector<int> output_ids;
  for (int i = 0; i < num_outputs; ++i)
    output_ids.push_back(i);
  return output_ids;
}

int Genome::get_num_neurons() const { return neurons.size(); }

std::vector<LinkGene>::iterator Genome::remove_link(const LinkId id_to_remove) {
  if (links.empty())
    return links.end();
  auto it =
      std::find_if(links.begin(), links.end(), [id_to_remove](LinkGene link) {
        return link.link_id.input_id == id_to_remove.input_id &&
               link.link_id.output_id == id_to_remove.output_id;
      });
  if (it != links.end()) {
    return links.erase(it);
  }
  return it;
}

void Genome::remove_hidden_neuron(const int neuron_id_to_remove) {
  if (!isHiddenNeuron(neuron_id_to_remove))
    return;

  auto it = std::find_if(neurons.begin(), neurons.end(),
                         [neuron_id_to_remove](const NeuronGene neuron) {
                           return neuron.neuron_id == neuron_id_to_remove;
                         });
  if (it != neurons.end()) {
    neurons.erase(it);
    for (auto it = links.begin(); it != links.end();) {
      LinkGene link = *it;
      if (link.link_id.input_id == neuron_id_to_remove ||
          link.link_id.output_id == neuron_id_to_remove)
        it = remove_link(link.link_id);
      else
        ++it;
    }
  }
}

int Genome::get_next_neuron_id() { return neurons.size() - num_inputs; }

bool Genome::isOutputNeuron(const int neuron_id) {
  return (0 <= neuron_id && neuron_id < num_outputs);
}

bool Genome::isInputNeuron(const int neuron_id) {
  return (-num_inputs <= neuron_id && neuron_id <= -1);
}

bool Genome::isHiddenNeuron(const int neuron_id) {
  return ((!isInputNeuron(neuron_id)) && (!isOutputNeuron(neuron_id)));
}
void Genome::set_link_weight(LinkId link_id, double new_weight) {
  auto it =
      std::find_if(links.begin(), links.end(), [link_id](const LinkGene &link) {
        return link_id.input_id == link.link_id.input_id &&
               link_id.output_id == link.link_id.output_id;
      });
  if (it != links.end()) {
    it->weight = new_weight; // Modifies the actual link in the vector
  }
}

void Genome::set_neuron_bias(int neuron_id, double new_bias) {
  auto it = std::find_if(neurons.begin(), neurons.end(),
                         [neuron_id](const NeuronGene &neuron) {
                           return neuron.neuron_id == neuron_id;
                         });
  if (it != neurons.end()) {
    it->bias = new_bias;
  }
}

void Genome::set_link_enabled(LinkId link_id, bool status) {
  auto it =
      std::find_if(links.begin(), links.end(), [link_id](const LinkGene &link) {
        return link_id.input_id == link.link_id.input_id &&
               link_id.output_id == link.link_id.output_id;
      });
  if (it != links.end()) {
    it->is_enabled = status; // Modifies the actual link in the vector
  }
}
