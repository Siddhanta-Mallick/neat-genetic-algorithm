#pragma once

struct NeatConfig
{
  int numGenerations = 1000;
  float survival_threshold = 0.25f;
  int num_inputs = 4;
  int num_outputs = 1;
  int population_size = 1000;
};

struct MutationConfig
{
  double min = -20.0;
  double max = 20.0;
  double mutation_rate = 0.2;
  double mutation_power = 1.2;
  double replacement_rate = 0.05;
};
