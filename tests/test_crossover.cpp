#include "../external/catch.hpp"

#include "Genotype.hpp"

#define private public
#include "Crossover.hpp"
#undef private

Rng rng_cross = Rng(1);
Indexer individualIndexer = Indexer();

Indexer &Crossover::genome_indexer_ = individualIndexer;
Rng &Crossover::rng_ = rng_cross;

TEST_CASE("test crossover_neuron") {

  int shared_id = 12;
  NeuronGene neuron1 = NeuronGene(shared_id, 6.9, ReLU());
  NeuronGene neuron2 = NeuronGene(shared_id, -4.20, Sigmoid());

  bool saw_neuron1_bias = false;
  bool saw_neuron2_bias = false;
  bool saw_neuron1_activation = false;
  bool saw_neuron2_activation = false;

  for (int i = 0; i < 100; i++) {
    NeuronGene child = Crossover::crossover_neuron(neuron1, neuron2);

    REQUIRE(child.neuron_id == shared_id);

    CHECK((child.bias == neuron1.bias || child.bias == neuron2.bias));

    CHECK((child.activation == neuron1.activation ||
           child.activation == neuron2.activation));

    if (child.bias == neuron1.bias)
      saw_neuron1_bias = true;
    if (child.bias == neuron2.bias)
      saw_neuron2_bias = true;
    if (child.activation == neuron1.activation)
      saw_neuron1_activation = true;
    if (child.activation == neuron2.activation)
      saw_neuron2_activation = true;
  }

  SECTION(
      "RNG effectively selects from both parents over multiple iterations") {
    CHECK(saw_neuron1_bias);
    CHECK(saw_neuron2_bias);
    CHECK(saw_neuron1_activation);
    CHECK(saw_neuron2_activation);
  }
}

TEST_CASE("test crossover_link") {

  LinkId shared_id = LinkId(12, -3);
  LinkGene link1(shared_id, 3.14, true);
  LinkGene link2(shared_id, -9.31, false);

  bool saw_link1_weight = false;
  bool saw_link2_weight = false;
  bool saw_link1_enabled = false;
  bool saw_link2_enabled = false;

  for (int i = 0; i < 100; i++) {
    LinkGene child = Crossover::crossover_link(link1, link2);

    REQUIRE(child.link_id == shared_id);

    CHECK((child.weight == link1.weight || child.weight == link2.weight));

    CHECK((child.is_enabled == link1.is_enabled ||
           child.is_enabled == link2.is_enabled));

    if (child.weight == link1.weight)
      saw_link1_weight = true;
    if (child.weight == link2.weight)
      saw_link2_weight = true;
    if (child.is_enabled == link1.is_enabled)
      saw_link1_enabled = true;
    if (child.is_enabled == link2.is_enabled)
      saw_link2_enabled = true;
  }

  SECTION(
      "RNG effectively selects from both parents over multiple iterations") {
    CHECK(saw_link1_weight);
    CHECK(saw_link2_weight);
    CHECK(saw_link1_enabled);
    CHECK(saw_link2_enabled);
  }
}

TEST_CASE("test crossover") {
  // 1. Common Setup (Mocks / Dependencies)

  int num_inputs = 2;  // IDs: 0, 1
  int num_outputs = 1; // IDs: -1

  // 2. Base Genomes Setup
  Genome g1(rng_cross, 1, num_inputs, num_outputs);
  Genome g2(rng_cross, 2, num_inputs, num_outputs);

  // Parent 1 (Will act as Dominant in most tests)
  g1.add_neuron(NeuronGene(2, 0.5, ReLU()));
  g1.add_link(LinkGene(LinkId(0, -1), 1.0, true)); // Matching link
  g1.add_link(LinkGene(LinkId(0, 2), 0.5, true));  // Disjoint dominant link
  g1.add_link(LinkGene(LinkId(2, -1), 0.5, true)); // Disjoint dominant link

  // Parent 2 (Will act as Recessive in most tests)
  g2.add_neuron(NeuronGene(3, -0.5, Sigmoid()));
  g2.add_link(LinkGene(LinkId(0, -1), -1.0, true)); // Matching link
  g2.add_link(LinkGene(LinkId(-1, 3), -0.5, true)); // Disjoint recessive link
  g2.add_link(LinkGene(LinkId(3, -1), -0.5, true)); // Disjoint recessive link

  Individual p1{g1, 100.0}; // High fitness
  Individual p2{g2, 50.0};  // Low fitness

  SECTION("Crossover routing correctly uses fitness to determine dominance") {
    Genome child1 = Crossover::crossover(p1, p2);

    // Child must inherit the disjoint neuron from dominant parent -> neuron
    // with id 2
    CHECK(child1.find_neuron(2).has_value());
    // Child must inherit the disjoint dominant link from g1, with linkid = (0,
    // 2)
    CHECK(child1.find_link(LinkId(0, 2)).has_value());

    // Child should NOT inherit p2's disjoint topology (Neuron 3)
    CHECK_FALSE(child1.find_neuron(3).has_value());
    CHECK_FALSE(child1.find_link(LinkId(-1, 3)).has_value());

    // reverse the fitness
    p1.fitness = 10.0;
    p2.fitness = 90.0;

    Genome child2 = Crossover::crossover(p1, p2);

    // Child should inherit p2's disjoint topology (Neuron 3)
    CHECK(child2.find_neuron(3).has_value());
    CHECK(child2.find_link(LinkId(-1, 3)).has_value());

    // Child should NOT inherit p1's disjoint topology (Neuron 2)
    CHECK_FALSE(child2.find_neuron(2).has_value());
    CHECK_FALSE(child2.find_link(LinkId(0, 2)).has_value());
  }

  SECTION(
      "crossover_individual handles matching and disjoint genes correctly") {
    // Explicitly calling crossover_individual to isolate its logic
    Genome child = Crossover::crossover_individual(p1, p2); // p1 is dominant

    // 1. Check matching genes (Homologous crossover)
    // Both parents had Link(0, -1). The child MUST have it.
    REQUIRE(child.find_link(LinkId(0, -1)).has_value());

    // Fetch the specific link to ensure its weight came from one of the parents
    std::optional<LinkGene> child_link = child.find_link(LinkId(0, -1));
    REQUIRE(child_link.has_value());
    CHECK((child_link->weight == 1.0 || child_link->weight == -1.0));

    // 2. Check disjoint/excess genes from Dominant (p1)
    // The child should inherit Neuron 2 and its links
    CHECK(child.find_neuron(2).has_value());
    CHECK(child.find_link(LinkId(0, 2)).has_value());
    CHECK(child.find_link(LinkId(2, -1)).has_value());

    // 3. Check disjoint/excess genes from Recessive (p2)
    // The child MUST NOT inherit Neuron 3 and its links
    CHECK_FALSE(child.find_neuron(3).has_value());
    CHECK_FALSE(child.find_link(LinkId(-1, 3)).has_value());
    CHECK_FALSE(child.find_link(LinkId(3, -1)).has_value());
  }

  SECTION("crossover_individual handles identical fitness edge case (Equal "
          "topology inheritance)") {
    // If fitness is perfectly equal, crossover(p1, p2) falls back to the `else`
    // branch (p2 dominant). Let's ensure the lengths are strictly bound by the
    // dominant parent.

    Genome child =
        Crossover::crossover_individual(p1, p1); // Crossing with self

    // Should have exactly the same number of neurons and links
    CHECK(child.get_neurons().size() == p1.genome.get_neurons().size());
    CHECK(child.get_links().size() == p1.genome.get_links().size());
  }
}
