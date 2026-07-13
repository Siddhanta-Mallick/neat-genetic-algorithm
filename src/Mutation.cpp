#include "Mutation.hpp"
#include "Activation.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"
#include <cstddef>
#include <optional>

int Mutation::choose_random_input_or_hidden(const Genome &genome) {
  int output_id_start = 0;
  int output_id_end = genome.get_num_outputs() - 1;

  int random_neuron_id = 0;
  do {
    auto random_neuron = rng_.choose_random_element(genome.get_neurons(),
                                                    genome.get_num_neurons());
    random_neuron_id = random_neuron.neuron_id;
  } while (output_id_start <= random_neuron_id &&
           random_neuron_id <= output_id_end);

  return random_neuron_id;
}

int Mutation::choose_random_output_or_hidden(const Genome &genome) {
  int input_id_start = -1;
  int input_id_end = -genome.get_num_inputs();

  int random_neuron_id = 0;
  do {
    auto random_neuron = rng_.choose_random_element(genome.get_neurons(),
                                                    genome.get_num_neurons());
    random_neuron_id = random_neuron.neuron_id;
  } while (input_id_end <= random_neuron_id &&
           random_neuron_id <= input_id_start);

  return random_neuron_id;
}

double Mutation::clamp(const double x) {
  return std::min(mutation_config_.max, std::max(x, mutation_config_.min));
}

LinkGene Mutation::mutate_new_link(const int input_id, const int output_id) {
  double weight = clamp(rng_.next_normal());
  return LinkGene{{input_id, output_id}, weight, true};
}

bool Mutation::mutate_add_link(Genome &genome) {
  int input_id = Mutation::choose_random_input_or_hidden(genome);
  int output_id = Mutation::choose_random_output_or_hidden(genome);

  if (CycleDetector::would_contain_cycle(genome.get_links(), input_id,
                                         output_id))
    return false;
  if (genome.find_link(LinkId(input_id, output_id)).has_value())
    return false;

  LinkGene newLink = Mutation::mutate_new_link(input_id, output_id);
  genome.add_link(newLink);
  return true;
}

void Mutation::mutate_remove_link(Genome &genome) {
  if (genome.get_links().empty())
    return;
  const LinkGene &link_to_remove =
      rng_.choose_random_element(genome.get_links(), genome.get_links().size());
  genome.remove_link(link_to_remove.link_id);
}

void Mutation::mutate_add_hidden_neuron(Genome &genome) {
  if (genome.get_links().empty())
    return;

  const LinkGene &link_to_split =
      rng_.choose_random_element(genome.get_links(), genome.get_links().size());
  const LinkGene link_to_split_copy = link_to_split;

  NeuronGene new_neuron =
      NeuronGene{genome.get_next_neuron_id(), rng_.next_normal(), ReLU{}};

  genome.add_neuron(new_neuron);
  genome.add_link(LinkGene{
      {link_to_split_copy.link_id.input_id, new_neuron.neuron_id}, 1.0, true});
  genome.add_link(
      LinkGene{{new_neuron.neuron_id, link_to_split_copy.link_id.output_id},
               link_to_split_copy.weight,
               true});
  // TODO - preserve the original link by only disabling, more in line with the
  // neat algorithm
  genome.remove_link(link_to_split_copy.link_id);
}

void Mutation::mutate_remove_hidden_neuron(Genome &genome) {
  if (genome.get_neurons().size() ==
      static_cast<size_t>(genome.get_num_inputs()) +
          static_cast<size_t>(genome.get_num_outputs()))
    return;
  NeuronGene random_neuron_to_remove;
  do {
    random_neuron_to_remove = rng_.choose_random_element(
        genome.get_neurons(), genome.get_neurons().size());
  } while (!genome.isHiddenNeuron(random_neuron_to_remove.neuron_id));

  genome.remove_hidden_neuron(random_neuron_to_remove.neuron_id);
}

double Mutation::mutate_add_delta(double original_value) {
  double delta = Mutation::clamp(
      rng_.next_normal(Mutation::mutation_config_.mutation_power));
  return Mutation::clamp(original_value + delta);
}

double Mutation::mutate_replace_val() {
  return Mutation::clamp(rng_.next_normal());
}

void Mutation::mutate(Genome &genome) {
  // using const neurongene reference to prevent headaches and of course
  // protecting the vector memory
  for (const NeuronGene &neuron : genome.get_neurons()) {
    if (rng_.next() < Mutation::mutation_config_.mutation_rate) {
      double new_bias = Mutation::mutate_add_delta(neuron.bias);
      genome.set_neuron_bias(neuron.neuron_id, new_bias);
    }

    if (rng_.next() < Mutation::mutation_config_.replacement_rate) {
      double new_bias = Mutation::mutate_replace_val();
      genome.set_neuron_bias(neuron.neuron_id, new_bias);
    }
  }

  for (const LinkGene &link : genome.get_links()) {
    if (rng_.next() < Mutation::mutation_config_.mutation_rate) {
      double new_weight = Mutation::mutate_add_delta(link.weight);
      genome.set_link_weight(link.link_id, new_weight);
    }

    if (rng_.next() < Mutation::mutation_config_.replacement_rate) {
      double new_weight = Mutation::mutate_replace_val();
      genome.set_link_weight(link.link_id, new_weight);
    }
  }

  // No need to panic, the following code does not invalidate the genome
  // reference because only vector references get invalidates
  if (rng_.next() < Mutation::mutation_config_.add_neuron_rate)
    Mutation::mutate_add_hidden_neuron(genome);

  if (rng_.next() < Mutation::mutation_config_.add_link_rate)
    Mutation::mutate_add_link(genome);

  if (rng_.next() < Mutation::mutation_config_.remove_neuron_rate)
    Mutation::mutate_remove_hidden_neuron(genome);

  if (rng_.next() < Mutation::mutation_config_.remove_link_rate)
    Mutation::mutate_remove_link(genome);
}
