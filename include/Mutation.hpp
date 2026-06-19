#pragma once

#include "Genotype.hpp"
#include "Utils.hpp"

class Mutation {
private:
  double clamp(double x) const;
  void mutate_add_link(Genome &genome);
  void mutate_remove_link(Genome &genome);
  void mutate_add_neuron(Genome &genome);
  void mutate_remove_neuron(Genome &genome);
  double mutate_new_value();
  double mutate_delta_value();
  int choose_random_input_or_hidden(const Genome &genome);
  int choose_random_output_or_hidden(const Genome &genome);

public:
  static Rng &rng_;
  static void mutate(Genome &genome);
};
