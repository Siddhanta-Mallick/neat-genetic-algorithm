#include "Mutation.hpp"

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
