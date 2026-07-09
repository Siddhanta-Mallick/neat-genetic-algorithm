#include "Genotype.hpp"
#include "Utils.hpp"
#include <../external/catch.hpp>
#include <cstddef>

#define private public
#include "Mutation.hpp"
#undef private

#include "Config.hpp"

Rng rng = Rng(1);

Rng &Mutation::rng_ = rng;
const MutationConfig &Mutation::mutation_config_ = MutationConfig{};

TEST_CASE("Test random neuron selection logic for mutation") {
  int id = 1;

  SECTION("choose_random_input_or_hidden") {

    SECTION("Small genome 3 inpputs, 1 output") {
      int num_inputs = 3;
      int num_outputs = 1;
      Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);
      int input_or_hidden_neuron_id =
          Mutation::choose_random_input_or_hidden(simple_genome);

      REQUIRE_FALSE((0 <= input_or_hidden_neuron_id &&
                     input_or_hidden_neuron_id <= num_outputs - 1));
    }

    SECTION("large genome 100 inputs, 25 outputs") {

      int num_inputs = 100;
      int num_outputs = 25;
      Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);

      int input_or_hidden_neuron_id =
          Mutation::choose_random_input_or_hidden(simple_genome);

      REQUIRE_FALSE((0 <= input_or_hidden_neuron_id &&
                     input_or_hidden_neuron_id <= num_outputs - 1));
    }
  }

  SECTION("choose_random_output_or_hidden") {

    SECTION("simple genome 3 inputs, 1 output") {
      int num_inputs = 3;
      int num_outputs = 1;
      Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);

      int output_or_hidden_neuron_id =
          Mutation::choose_random_output_or_hidden(simple_genome);
      bool is_hidden_neuron_id =
          (1 <= output_or_hidden_neuron_id &&
           output_or_hidden_neuron_id <= simple_genome.get_num_neurons() - 1);
      bool is_output_neuron_id =
          (0 <= output_or_hidden_neuron_id &&
           output_or_hidden_neuron_id <= simple_genome.get_num_outputs() - 1);
      REQUIRE((is_hidden_neuron_id || is_output_neuron_id));
    }

    SECTION("large genome 100 inputs, 25 outputs") {

      int num_inputs = 100;
      int num_outputs = 25;
      Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);

      int output_or_hidden_neuron_id =
          Mutation::choose_random_output_or_hidden(simple_genome);
      bool is_hidden_neuron_id =
          (1 <= output_or_hidden_neuron_id &&
           output_or_hidden_neuron_id <= simple_genome.get_num_neurons() - 1);
      bool is_output_neuron_id =
          (0 <= output_or_hidden_neuron_id &&
           output_or_hidden_neuron_id <= simple_genome.get_num_outputs() - 1);
      REQUIRE((is_hidden_neuron_id || is_output_neuron_id));
    }
  }
}

TEST_CASE("Test mutate_add_link") {
  int id = 1;
  int num_inputs = 3;
  int num_outputs = 1;
  Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);

  size_t size_before_mutation = simple_genome.get_links().size();
  bool success = Mutation::mutate_add_link(simple_genome);
  bool addLinkSuccess = false;
  if (success)
    addLinkSuccess =
        (size_before_mutation + 1 == simple_genome.get_links().size());
  else
    addLinkSuccess = (size_before_mutation == simple_genome.get_links().size());
  REQUIRE(addLinkSuccess == true);
}

TEST_CASE("Test mutate_remove_link") {
  int id = 1;
  int num_inputs = 3;
  int num_outputs = 1;
  Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);

  size_t countBefore = simple_genome.get_links().size();
  Mutation::mutate_remove_link(simple_genome);
  size_t countAfter = simple_genome.get_links().size();
  REQUIRE(countBefore - 1 == countAfter);
}

TEST_CASE("Test mutate_add_neuron") {
  int id = 1;
  int num_inputs = 3;
  int num_outputs = 1;
  Genome simple_genome = Genome(rng, id, num_inputs, num_outputs);

  size_t neuronsBefore = simple_genome.get_neurons().size();
  size_t linksBefore = simple_genome.get_links().size();
  Mutation::mutate_add_hidden_neuron(simple_genome);

  // testing logic is adding a new nerurons splits existing link, thereby the
  // genome having one new neuron and one new link (along with the exsting link)

  size_t neuronsAfter = simple_genome.get_neurons().size();
  size_t linksAfter = simple_genome.get_links().size();

  REQUIRE(neuronsBefore + 1 == neuronsAfter);
  REQUIRE(linksBefore + 1 == linksAfter);
}

TEST_CASE("Test mutate_remove_hidden_neuron") {
  int genome_id = 0;
  int num_inputs = 3;
  int num_outputs = 1;
  Genome simple_genome = Genome(rng, genome_id, num_inputs, num_outputs);

  simple_genome.add_neuron(
      NeuronGene(simple_genome.get_next_neuron_id(), 0, ReLU()));

  // 3 links associated with the new hidden neuron
  simple_genome.add_link(LinkGene(LinkId(-1, 1), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(-3, 1), 0.0, true));
  simple_genome.add_link(LinkGene(LinkId(1, 0), 0.0, true));

  int neuron_count_before = simple_genome.get_neurons().size();
  int link_count_before = simple_genome.get_links().size();

  Mutation::mutate_remove_hidden_neuron(simple_genome);

  int neuron_count_after = simple_genome.get_neurons().size();
  int link_count_after = simple_genome.get_links().size();

  REQUIRE(neuron_count_before - 1 == neuron_count_after);
  REQUIRE(link_count_before - 3 == link_count_after);
}

TEST_CASE("Test mutate_add_delta") {
  double original_value = rng.next();
  double delta_mutated_value = Mutation::mutate_add_delta(original_value);
  REQUIRE_FALSE(original_value == delta_mutated_value);
}

TEST_CASE("Test mutate_replace_value") {
  double original_value = rng.next();
  double replaced_value = Mutation::mutate_replace_val();
  REQUIRE_FALSE(original_value == replaced_value);
}

TEST_CASE("Test mutate") {
  int genome_id = 0;
  int num_inputs = 3;
  int num_outputs = 1;
  Genome simple_genome = Genome(rng, genome_id, num_inputs, num_outputs);
  REQUIRE_NOTHROW(Mutation::mutate(simple_genome));
}
