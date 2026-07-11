#include "FeedForwardNeuralNetwork.hpp"
#include <algorithm>
#include <assert.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(
    std::vector<int> inputs, std::vector<int> outputs,
    std::vector<Neuron> neurons) {
  input_ids = inputs;
  output_ids = outputs;
  this->neurons = neurons;
}

std::vector<std::vector<int>>
FeedForwardNeuralNetwork::toposort_neurons_into_layers(
    const std::vector<int> inputs, const std::vector<int> outputs,
    const std::vector<LinkGene> &links) {

  std::vector<std::vector<int>> layers;
  std::unordered_map<int, std::vector<int>> adj;
  std::unordered_map<int, int> indegree;
  std::unordered_set<int> all_nodes;

  for (int id : inputs)
    all_nodes.insert(id);
  for (int id : outputs)
    all_nodes.insert(id);

  for (const auto &link : links) {
    if (link.is_enabled) {
      all_nodes.insert(link.link_id.input_id);
      all_nodes.insert(link.link_id.output_id);
      adj[link.link_id.input_id].push_back(link.link_id.output_id);
      indegree[link.link_id.output_id]++;
    }
  }

  std::queue<int> q;
  for (int node_id : all_nodes) {
    if (indegree[node_id] == 0)
      q.push(node_id);
  }

  while (!q.empty()) {
    int q_length = q.size();
    std::vector<int> layer;

    for (int i = 0; i < q_length; ++i) {
      int node = q.front();
      q.pop();
      layer.push_back(node);

      if (adj.contains(node)) {
        for (int neighbor : adj[node]) {
          indegree[neighbor]--;
          if (indegree[neighbor] == 0) {
            q.push(neighbor);
          }
        }
      }
    }
    layers.push_back(layer);
  }

  return layers;
}

FeedForwardNeuralNetwork
FeedForwardNeuralNetwork::create_nn_from_gene(const Genome &genome) {
  std::vector<int> inputs = genome.get_input_ids();
  std::vector<int> outputs = genome.get_output_ids();
  std::vector<std::vector<int>> layers =
      FeedForwardNeuralNetwork::toposort_neurons_into_layers(
          inputs, outputs, genome.get_links());

  std::vector<Neuron> neurons;

  for (const auto &layer : layers) {
    for (int neuron_id : layer) {
      std::vector<NeuronInput> neuron_inputs;
      for (auto link : genome.get_links())
        if (neuron_id == link.link_id.output_id)
          neuron_inputs.emplace_back(
              NeuronInput{link.link_id.input_id, link.weight});
      std::optional<NeuronGene> neuron_gene_opt = genome.find_neuron(neuron_id);
      assert(neuron_gene_opt.has_value());
      neurons.emplace_back(
          Neuron{(*neuron_gene_opt).neuron_id, (*neuron_gene_opt).bias,
                 std::move(neuron_inputs), (*neuron_gene_opt).activation});
    }
  }
  return FeedForwardNeuralNetwork(std::move(inputs), std::move(outputs),
                                  std::move(neurons));
}

std::vector<double>
FeedForwardNeuralNetwork::activate(const std::vector<double> inputs) {
  assert(inputs.size() == input_ids.size());

  std::unordered_map<int, double> values;

  for (size_t i = 0; i < inputs.size(); ++i) {
    int input_id = input_ids[i];
    values[input_id] = inputs[i];
  }

  for (const auto &neuron : neurons) {
    double z = 0.0;
    for (auto input : neuron.inputs) {
      assert(values.contains(input.input_id));
      z += input.weight * values[input.input_id];
    }
    z += neuron.bias;
    double a = std::visit(ActivationFn{z}, neuron.activation);
    values[neuron.neuron_id] = a;
  }

  std::vector<double> outputs;
  for (int output_id : output_ids) {
    assert(values.contains(output_id));
    outputs.push_back(values[output_id]);
  }

  return outputs;
}
