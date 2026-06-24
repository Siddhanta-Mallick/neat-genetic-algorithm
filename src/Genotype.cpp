#include "Genotype.hpp"
#include "Utils.hpp"

Genome::Genome(Rng &rng, int id, int inputs, int outputs)
    : rng_(rng), genome_id(id), num_inputs(inputs), num_outputs(outputs) {
  for (int i = 0; i < num_inputs; i++)
    neurons.push_back(NeuronGene{-i - 1, rng_.next(), ReLU{}});
  for (int i = 0; i < num_outputs; i++)
    neurons.push_back(NeuronGene{i, rng_.next(), Sigmoid{}});

  for (int i = 0; i < num_inputs; ++i) {
    int input_id = -i - 1;
    for (int output_id = 0; output_id < num_outputs; ++output_id) {
      links.push_back(LinkGene{LinkId{input_id, output_id}, rng_.next(), true});
    }
  }
}

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

void Genome::add_neuron(const NeuronGene newNeuron) {
  neurons.push_back(newNeuron);
}

void Genome::add_link(const LinkGene newLink) { links.push_back(newLink); }

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

void Genome::remove_link(const LinkGene &link_to_remove) {
  if (links.empty())
    return;

  std::ptrdiff_t index = &link_to_remove - &links[0];
  if (index >= 0 && static_cast<size_t>(index) < links.size())
    links.erase(links.begin() + index);
}

int Genome::get_next_neuron_id() { return neurons.size() - num_inputs; }
