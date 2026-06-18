#pragma once

#include "Genotype.hpp"

class Mutation {
private:
  double clamp(double x) const;
  void mutate_add_link(Genome &genome);
  void mutate_remove_link(Genome &genome);
  void mutate_add_neuron(Genome &genome);
  void mutate_remove_neuron(Genome &genome);
  double mutate_new_value();
  double mutate_delta_value();

public:
  static void mutate(Genome &genome);
};
