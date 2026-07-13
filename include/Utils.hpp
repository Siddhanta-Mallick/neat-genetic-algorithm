#pragma once

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "Genotype.hpp"

class Rng {
private:
  double min_;
  double max_;
  std::random_device rd_;
  std::mt19937 generator_;
  std::uniform_real_distribution<double> distribution_;
  std::normal_distribution<double> normal_dist_;

public:
  Rng(std::optional<long> seed, double min = 0.0, double max = 1.0)
      : min_(min), max_(max),
        generator_(seed.has_value() ? std::mt19937(*seed)
                                    : std::mt19937(rd_())),
        distribution_(min_, max_) {}

  double next() { return distribution_(generator_); }

  double next_normal(double std_dev = 1) {
    return normal_dist_(generator_) * std_dev;
  }

  template <typename T> T choose(const T &a, const T &b) {
    if (next() < 0.5)
      return a;
    else
      return b;
  }

  template <typename Element>
  const Element &choose_random_element(const std::vector<Element> &arr,
                                       int cutoff) {
    int random_index = std::floor(next() * cutoff);
    assert(random_index >= 0 && random_index < cutoff);
    return arr[random_index];
  }
};

class Indexer {
private:
  int next_id_;

public:
  Indexer(int next_id = 0) : next_id_(next_id) {};
  int next() { return next_id_++; }
};

class CycleDetector {
public:
  static bool dfs(int node, std::unordered_map<int, std::vector<int>> &graph,
                  std::unordered_map<int, int> &state) {
    if (state[node] == 1)
      return true;
    if (state[node] == 2)
      return false;

    state[node] = 1;

    for (int neighbour : graph[node])
      if (dfs(neighbour, graph, state))
        return true;

    state[node] = 2;
    return false;
  }

  static bool would_contain_cycle(const std::vector<LinkGene> &links, int input,
                                  int output) {
    std::unordered_map<int, std::vector<int>> graph;

    for (const LinkGene link : links)
      graph[link.link_id.input_id].push_back(link.link_id.output_id);

    graph[input].push_back(output);

    std::unordered_map<int, int> state;

    for (const auto &pair : graph)
      // Values are auto initialized to 0 if key does not exist. No need for a
      // default 0 val.
      if (state[pair.first] == 0)
        if (dfs(pair.first, graph, state))
          return true;

    return false;
  }
};
