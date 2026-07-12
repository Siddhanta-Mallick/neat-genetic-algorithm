#pragma once

struct NeatConfig {
  int numGenerations = 1000;
  float survival_threshold = 0.25f;
  int num_inputs = 4;
  int num_outputs = 1;
  int population_size = 1000;
};

struct MutationConfig {
  double min = -20.0;
  double max = 20.0;
  const double mutation_probability = 0.05;
  double mutation_rate = 0.2;
  double mutation_power = 1.2;
  double replacement_rate = 0.05;
  const double add_link_rate = 0.05;
  const double add_neuron_rate = 0.03;
  const double remove_link_rate = 0.005;
  const double remove_neuron_rate = 0.001;
};
