#include <assert.h>
#include <optional>

#include "Activation.hpp"
#include "Crossover.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"

NeuronGene Crossover::crossover_neuron(const NeuronGene &a, NeuronGene &b) {
  assert(a.neuron_id == b.neuron_id);
  int neuron_id = a.neuron_id;
  double bias = rng_.choose(a.bias, b.bias);
  Activation activation = rng_.choose(a.activation, b.activation);
  return {neuron_id, bias, activation};
}

LinkGene Crossover::crossover_link(const LinkGene &a, LinkGene &b) {
  assert(a.link_id.input_id == b.link_id.input_id &&
         a.link_id.output_id == b.link_id.output_id);
  LinkId link_id = a.link_id;
  double weight = rng_.choose(a.weight, b.weight);
  bool is_enabled = rng_.choose(a.is_enabled, b.is_enabled);
  return {link_id, weight, is_enabled};
}

Genome Crossover::crossover(const Individual &p1, const Individual &p2) {
  if (p1.fitness > p2.fitness)
    return crossover_individual(p1, p2);
  else
    return crossover_individual(p2, p1);
}

Genome Crossover::crossover_individual(const Individual &dominant,
                                       const Individual &recessive) {

  Genome offspring(rng_, genome_indexer_.next());

  // Iherit the neurons - cross over matching neurons otherwise select the
  // dominant one
  for (const auto &dominant_neuron : dominant.genome.get_neurons()) {

    int neuron_id = dominant_neuron.neuron_id;
    std::optional<NeuronGene> recessive_neuron =
        recessive.genome.find_neuron(neuron_id);

    if (recessive_neuron.has_value())
      offspring.add_neuron(
          crossover_neuron(dominant_neuron, *recessive_neuron));
    else
      offspring.add_neuron(dominant_neuron);
  }

  // Inherit the link genes - cross over matching links, otherwise copy the
  // disjoint link from dominant
  for (const auto &dominant_link : dominant.genome.get_links()) {
    LinkId link_id = dominant_link.link_id;
    std::optional<LinkGene> recessive_link =
        recessive.genome.find_link(link_id);
    if (recessive_link.has_value())
      offspring.add_link(crossover_link(dominant_link, *recessive_link));
    else
      offspring.add_link(dominant_link);
  }
  return offspring;
}
