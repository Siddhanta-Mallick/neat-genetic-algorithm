#include "../external/catch.hpp"
#include "Utils.hpp"

#define private public
#include "FeedForwardNeuralNetwork.hpp"
#undef private

// Global mock RNG for deterministic tests
Rng rng_ffn = Rng(1);
double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

TEST_CASE("Topological Sorting of Neurons", "[FeedForwardNeuralNetwork]") {
  SECTION("Linear chain network") {
    std::vector<int> inputs = {-1};
    std::vector<int> outputs = {1};
    // Input (-1) -> Hidden (0) -> Output (1)
    std::vector<LinkGene> links = {{{-1, 0}, 1.0, true}, {{0, 1}, 1.0, true}};

    auto layers = FeedForwardNeuralNetwork::toposort_neurons_into_layers(
        inputs, outputs, links);

    REQUIRE(layers.size() == 3);
    REQUIRE(layers[0] == std::vector<int>{-1});
    REQUIRE(layers[1] == std::vector<int>{0});
    REQUIRE(layers[2] == std::vector<int>{1});
  }

  SECTION("Handles disconnected output nodes") {
    std::vector<int> inputs = {-1};
    std::vector<int> outputs = {0, 1}; // Node 1 has no incoming links
    std::vector<LinkGene> links = {{{-1, 0}, 1.0, true}};

    auto layers = FeedForwardNeuralNetwork::toposort_neurons_into_layers(
        inputs, outputs, links);

    // Node -1 and Node 1 both have in-degree 0, they should be in the first
    // layer.
    REQUIRE(layers.size() == 2);

    bool has_node_minus_1 =
        std::find(layers[0].begin(), layers[0].end(), -1) != layers[0].end();
    bool has_node_1 =
        std::find(layers[0].begin(), layers[0].end(), 1) != layers[0].end();

    REQUIRE(has_node_minus_1 == true);
    REQUIRE(has_node_1 == true);
    REQUIRE(layers[1] == std::vector<int>{0});
  }
}
TEST_CASE("Neural Network Creation and Activation",
          "[FeedForwardNeuralNetwork]") {

  SECTION("Base fully-connected calculation (No hidden nodes)") {
    int num_inputs = 2;
    int num_outputs = 1;

    Genome genome(rng_ffn, 1, num_inputs, num_outputs);

    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(-2, 0.0);
    genome.set_neuron_bias(0, 1.0);

    genome.set_link_weight(LinkId{-1, 0}, 2.0);
    genome.set_link_weight(LinkId{-2, 0}, -1.0);

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {2.0, 3.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Z = (2.0 * 2.0) + (3.0 * -1.0) + 1.0 = 2.0
    // Expected: Sigmoid(2.0)
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(sigmoid(2.0)));
  }

  SECTION("Complex topology with hidden node and disabled links") {
    int num_inputs = 2;
    int num_outputs = 1;

    Genome genome(rng_ffn, 2, num_inputs, num_outputs);

    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(-2, 0.0);
    genome.set_neuron_bias(0, 1.0);

    genome.set_link_weight(LinkId{-2, 0}, -1.0);
    genome.set_link_enabled(LinkId{-1, 0}, false);

    // Hidden Node explicitly uses ReLU
    genome.add_neuron(NeuronGene{1, 0.5, ReLU{}});

    genome.add_link(LinkGene{LinkId{-1, 1}, 2.0, true});
    genome.add_link(LinkGene{LinkId{1, 0}, 1.5, true});

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {2.0, 3.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Hidden Node(1) Input: (2.0 * 2.0) + 0.5 = 4.5. ReLU(4.5) = 4.5
    // Output Node(0) Z: (4.5 * 1.5) + (3.0 * -1.0) + 1.0 = 4.75.
    // Expected: Sigmoid(4.75)
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(sigmoid(4.75)));
  }

  SECTION("Disabled links are strictly ignored") {
    int num_inputs = 1;
    int num_outputs = 1;

    Genome genome(rng_ffn, 3, num_inputs, num_outputs);

    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(0, 0.5);

    genome.set_link_weight(LinkId{-1, 0}, 5.0);
    genome.set_link_enabled(LinkId{-1, 0}, false);

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {2.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Z = 0.5 (Bias only, because the 5.0 weight link is disabled)
    // Expected: Sigmoid(0.5)
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(sigmoid(0.5)));
  }

  SECTION("Sigmoid handles negative bounds properly") {
    int num_inputs = 1;
    int num_outputs = 2;

    Genome genome(rng_ffn, 4, num_inputs, num_outputs);

    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(0, 0.0);
    genome.set_neuron_bias(1, 0.0);

    genome.set_link_weight(LinkId{-1, 0}, 1.0);
    genome.set_link_weight(LinkId{-1, 1}, -5.0);

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {2.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Output 0 Z: (2.0 * 1.0) = 2.0
    // Output 1 Z: (2.0 * -5.0) = -10.0
    REQUIRE(output_vals.size() == 2);
    REQUIRE(output_vals[0] == Approx(sigmoid(2.0)));
    REQUIRE(output_vals[1] == Approx(sigmoid(-10.0)));
  }

  // =====================================================================
  // EXTENSIVE EDGE CASES AND DEEP TOPOLOGY TESTS
  // =====================================================================

  SECTION("Negative raw inputs pass through correctly without being clamped") {
    int num_inputs = 1;
    int num_outputs = 1;

    Genome genome(rng_ffn, 5, num_inputs, num_outputs);
    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(0, 0.0);

    // Direct link from Input to Output
    genome.set_link_weight(LinkId{-1, 0}, 1.0);

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    // Pass a heavily negative input
    std::vector<double> input_vals = {-5.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // If input was clamped by ReLU, Z would be 0.
    // Since it acts as a pass-through, Z = -5.0 * 1.0 = -5.0
    // Expected: Sigmoid(-5.0)
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(sigmoid(-5.0)));
  }

  SECTION("Hidden nodes strictly clamp negative internal sums to zero (ReLU "
          "behavior)") {
    int num_inputs = 1;
    int num_outputs = 1;

    Genome genome(rng_ffn, 6, num_inputs, num_outputs);
    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(0, 0.0);

    // Disable direct routing
    genome.set_link_enabled(LinkId{-1, 0}, false);

    // Add a hidden node (ID: 1) with 0.0 bias
    genome.add_neuron(NeuronGene{1, 0.0, ReLU{}});

    // Input(-1) -> Hidden(1) -> Output(0)
    // We give the first link a negative weight to force a negative sum
    genome.add_link(LinkGene{LinkId{-1, 1}, -2.0, true});
    genome.add_link(LinkGene{LinkId{1, 0}, 5.0, true});

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {3.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Hidden(1) Input: (3.0 * -2.0) + 0.0 = -6.0
    // Hidden(1) Activation: ReLU(-6.0) = 0.0  <-- CLAMPED!
    // Output(0) Z: (0.0 * 5.0) + 0.0 = 0.0
    // Expected: Sigmoid(0.0) = 0.5
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(0.5));
  }

  SECTION(
      "Deep architecture propagates correctly through multiple hidden layers") {
    int num_inputs = 1;
    int num_outputs = 1;

    Genome genome(rng_ffn, 7, num_inputs, num_outputs);
    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(0, 0.5);
    genome.set_link_enabled(LinkId{-1, 0}, false); // Disable direct link

    // Create a chain of hidden nodes: Input(-1) -> Hidden(1) -> Hidden(2) ->
    // Output(0)
    genome.add_neuron(NeuronGene{1, 1.0, ReLU{}});
    genome.add_neuron(NeuronGene{2, -2.0, ReLU{}});

    genome.add_link(LinkGene{LinkId{-1, 1}, 2.0, true});
    genome.add_link(LinkGene{LinkId{1, 2}, 3.0, true});
    genome.add_link(LinkGene{LinkId{2, 0}, -1.0, true});

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {1.5};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Math Breakdown:
    // Hidden(1) In: (1.5 * 2.0) + 1.0(bias) = 4.0. ReLU(4.0) = 4.0
    // Hidden(2) In: (4.0 * 3.0) - 2.0(bias) = 10.0. ReLU(10.0) = 10.0
    // Output(0) Z: (10.0 * -1.0) + 0.5(bias) = -9.5
    // Expected: Sigmoid(-9.5)
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(sigmoid(-9.5)));
  }

  SECTION("Multi-path routing (ResNet style) combines sums accurately") {
    int num_inputs = 1;
    int num_outputs = 1;

    Genome genome(rng_ffn, 8, num_inputs, num_outputs);
    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(0, 0.0);

    // Keep the direct link enabled this time (Skip connection)
    genome.set_link_weight(LinkId{-1, 0}, 0.5);

    // Add a hidden node path running in parallel
    genome.add_neuron(NeuronGene{1, 0.0, ReLU{}});
    genome.add_link(LinkGene{LinkId{-1, 1}, 2.0, true});
    genome.add_link(LinkGene{LinkId{1, 0}, 1.5, true});

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    std::vector<double> input_vals = {2.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Math Breakdown:
    // Path 1 (Hidden): (2.0 * 2.0) + 0.0 = 4.0 -> ReLU(4.0) = 4.0
    // Path 2 (Direct): 2.0 * 0.5 = 1.0
    // Output(0) Z calculates from both incoming links:
    // Z = (4.0 * 1.5)[from Hidden] + (2.0 * 0.5)[from Input] + 0.0(bias)
    // Z = 6.0 + 1.0 = 7.0
    // Expected: Sigmoid(7.0)
    REQUIRE(output_vals.size() == 1);
    REQUIRE(output_vals[0] == Approx(sigmoid(7.0)));
  }

  SECTION("Zeros across the board evaluate properly") {
    int num_inputs = 2;
    int num_outputs = 2;

    Genome genome(rng_ffn, 9, num_inputs, num_outputs);

    // Everything is 0
    genome.set_neuron_bias(-1, 0.0);
    genome.set_neuron_bias(-2, 0.0);
    genome.set_neuron_bias(0, 0.0);
    genome.set_neuron_bias(1, 0.0);

    genome.set_link_weight(LinkId{-1, 0}, 0.0);
    genome.set_link_weight(LinkId{-2, 0}, 0.0);
    genome.set_link_weight(LinkId{-1, 1}, 0.0);
    genome.set_link_weight(LinkId{-2, 1}, 0.0);

    auto nn = FeedForwardNeuralNetwork::create_nn_from_gene(genome);

    // Input is 0
    std::vector<double> input_vals = {0.0, 0.0};
    std::vector<double> output_vals = nn.activate(input_vals);

    // Z = 0.0 for everything
    // Sigmoid(0.0) = 0.5
    REQUIRE(output_vals.size() == 2);
    REQUIRE(output_vals[0] == Approx(0.5));
    REQUIRE(output_vals[1] == Approx(0.5));
  }
}
