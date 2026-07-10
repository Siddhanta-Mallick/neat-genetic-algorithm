#include "../external/catch.hpp"
#include "Genotype.hpp"
#include "Utils.hpp"
#include <cstddef>

Rng rng_genome(1);

TEST_CASE("Test Constructor of genome") {
  int genome_id = 0;
  int num_inputs = 10;
  int num_outputs = 5;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  REQUIRE(simple_genome.get_genome_id() == genome_id);
  REQUIRE(simple_genome.get_num_inputs() == num_inputs);
  REQUIRE(simple_genome.get_num_outputs() == num_outputs);
  REQUIRE(simple_genome.get_num_neurons() == num_inputs + num_outputs);
}

TEST_CASE("Test find_neuron simple_genome") {
  int genome_id = 0;
  int num_inputs = 10;
  int num_outputs = 5;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  REQUIRE(simple_genome.find_neuron(0).has_value() == true);
  REQUIRE(simple_genome.find_neuron(4).has_value() == true);
  REQUIRE(simple_genome.find_neuron(5).has_value() == false);

  REQUIRE(simple_genome.find_neuron(-1).has_value() == true);
  REQUIRE(simple_genome.find_neuron(-10).has_value() == true);
  REQUIRE(simple_genome.find_neuron(-11).has_value() == false);
}

TEST_CASE("Test find_link simple_genome") {
  int genome_id = 0;
  int num_inputs = 10;
  int num_outputs = 5;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  REQUIRE(simple_genome.find_link(LinkId(-1, 0)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(-1, 4)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(-1, 5)).has_value() == false);

  REQUIRE(simple_genome.find_link(LinkId(-10, 0)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(-10, 4)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(-10, 5)).has_value() == false);
}

TEST_CASE("Test add_neuron on simple_genome") {
  int genome_id = 0;
  int num_inputs = 10;
  int num_outputs = 5;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  int prevCount = simple_genome.get_neurons().size();
  int newNeuronId = simple_genome.get_next_neuron_id();

  simple_genome.add_neuron(
      NeuronGene(simple_genome.get_next_neuron_id(), 0, ReLU()));

  REQUIRE(simple_genome.get_neurons().size() ==
          static_cast<size_t>(prevCount + 1));
  REQUIRE(simple_genome.find_neuron(newNeuronId).has_value() == true);

  // Test with invalid neuron id
  prevCount = simple_genome.get_neurons().size();
  int requiredId = simple_genome.get_next_neuron_id();
  simple_genome.add_neuron(NeuronGene(requiredId, 0, ReLU()));

  REQUIRE(simple_genome.get_neurons().size() ==
          static_cast<size_t>(prevCount + 1));
  REQUIRE(simple_genome.find_neuron(requiredId).has_value() == true);
}

TEST_CASE("Test add_link") {
  int genome_id = 0;
  int num_inputs = 10;
  int num_outputs = 5;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  simple_genome.add_neuron(
      NeuronGene(simple_genome.get_next_neuron_id(), 0, ReLU()));

  // test by adding valid links
  simple_genome.add_link(LinkGene(LinkId(-1, 5), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(-10, 5), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(5, 0), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(5, 4), 0.0, true));

  REQUIRE(simple_genome.find_link(LinkId(-1, 5)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(-10, 5)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(5, 0)).has_value() == true);
  REQUIRE(simple_genome.find_link(LinkId(5, 4)).has_value() == true);

  // test by adding an invalid link
  simple_genome.add_link(LinkGene(LinkId(-20, 0), 0.0, true));
  REQUIRE(simple_genome.find_link(LinkId(-20, 0)).has_value() == false);
}

TEST_CASE("Test remove_link") {
  int genome_id = 0;
  int num_inputs = 10;
  int num_outputs = 5;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  int link_count_before = simple_genome.get_links().size();
  simple_genome.remove_link(LinkId(-1, 0));
  int link_count_after = simple_genome.get_links().size();

  REQUIRE(link_count_before - 1 == link_count_after);
}

TEST_CASE("Test remove_hideen_neuron") {
  int genome_id = 0;
  int num_inputs = 3;
  int num_outputs = 1;
  Genome simple_genome = Genome(rng_genome, genome_id, num_inputs, num_outputs);

  simple_genome.add_neuron(
      NeuronGene(simple_genome.get_next_neuron_id(), 0, ReLU()));

  // 3 links associated with the new hidden neuron
  simple_genome.add_link(LinkGene(LinkId(-1, 1), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(-3, 1), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(1, 0), 0.0, true));

  int neuron_count_before = simple_genome.get_neurons().size();
  int link_count_before = simple_genome.get_links().size();

  simple_genome.remove_hidden_neuron(1);

  int neuron_count_after = simple_genome.get_neurons().size();
  int link_count_after = simple_genome.get_links().size();

  REQUIRE(neuron_count_before - 1 == neuron_count_after);
  REQUIRE(link_count_before - 3 == link_count_after);
}
