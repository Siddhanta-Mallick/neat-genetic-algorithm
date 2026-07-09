#pragma once

#include "Config.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"

struct LinkGene;

class Mutation {
private:
  static int choose_random_input_or_hidden(const Genome &genome);
  static int choose_random_output_or_hidden(const Genome &genome);
  static const MutationConfig &mutation_config_;

  // 4 Structural Mutations
  static bool mutate_add_link(Genome &genome);
  static void mutate_remove_link(Genome &genome);
  static void mutate_add_hidden_neuron(Genome &genome);
  static void mutate_remove_hidden_neuron(Genome &genome);

  // Non Structural Mutations
  static double mutate_add_delta(double original_value);
  static double mutate_replace_val();

  // Helpers
  static LinkGene mutate_new_link(const int input_id, const int output_id);
  static double clamp(const double x);

public:
  static Rng &rng_;
  static void mutate(Genome &genome);
};
