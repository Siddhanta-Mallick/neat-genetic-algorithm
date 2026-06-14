#include "FeedForwardNeuralNetwork.hpp"
#include <queue>
#include <unordered_map>
#include <vector>

std::vector<std::vector<int>>
FeedForwardNeuralNetwork::toposort_neurons_into_layers(
    const std::vector<int> inputs, const std::vector<int> outputs,
    const std::vector<LinkGene> &links) {

  std::vector<std::vector<int>> layers;
  std::unordered_map<int, std::vector<int>> adj;
  std::unordered_map<int, int> indegree;

  for (const auto &link : links) {
    if (link.is_enabled) {
      adj[link.link_id.input_id].push_back(link.link_id.output_id);
      indegree[link.link_id.output_id]++;
    }
  }

  std::queue<int> q;
  for (int input_id : inputs) {
    q.push(input_id);
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
