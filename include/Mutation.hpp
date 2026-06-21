#pragma once

#include "Config.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"

class Mutation {
private:
  double clamp(double x) const;
  void mutate_add_link(Genome &genome);
  void mutate_remove_link(Genome &genome);
  void mutate_add_neuron(Genome &genome);
  void mutate_remove_neuron(Genome &genome);
  double mutate_delta_value();
  int choose_random_input_or_hidden(const Genome &genome);
  int choose_random_output_or_hidden(const Genome &genome);

public:
  static Rng &rng_;
  static void mutate(Genome &genome);
};

class LinkMutator {
private:
  static Rng &rng_;
  static const MutationConfig &mutation_config_;

public:
  static LinkGene new_value(const int input_id, const int output_id);
  static double clamp(const double x);
};
