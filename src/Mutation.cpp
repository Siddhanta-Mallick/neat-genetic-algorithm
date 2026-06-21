#include "Mutation.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"

int Mutation::choose_random_input_or_hidden(const Genome &genome) {
  int output_id_start = 0;
  int output_id_end = genome.get_num_ouputs() - 1;

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

double LinkMutator::clamp(const double x) {
  return std::min(mutation_config_.max, std::max(x, mutation_config_.min));
}

LinkGene LinkMutator::new_value(const int input_id, const int output_id) {
  double weight = clamp(rng_.next());
  return LinkGene{{input_id, output_id}, weight, true};
}

void Mutation::mutate_add_link(Genome &genome) {
  int input_id = choose_random_input_or_hidden(genome);
  int output_id = choose_random_output_or_hidden(genome);

  if (CycleDetector::would_contain_cycle(genome.get_links(), input_id,
                                         output_id, genome.get_num_neurons(),
                                         genome.get_num_inputs()))
    return;

  LinkGene newLink = LinkMutator::new_value(input_id, output_id);
  genome.add_link(newLink);
}
