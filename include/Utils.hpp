#pragma once

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <optional>
#include <random>
#include <vector>

class Rng {
private:
  double min_;
  double max_;
  std::random_device rd_;
  std::mt19937 generator_;
  std::uniform_real_distribution<double> distribution_;

public:
  Rng(std::optional<long> seed, double min = 0.0, double max = 1.0)
      : min_(min), max_(max),
        generator_(seed.has_value() ? std::mt19937(*seed)
                                    : std::mt19937(rd_())),
        distribution_(min_, max_) {}

  double next() { return distribution_(generator_); }

  template <typename T> T choose(const T &a, const T &b) {
    if (next() < 0.5)
      return a;
    else
      return b;
  }

  template <typename Element>
  Element &choose_random_element(const std::vector<Element> &arr, int cutoff) {
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
