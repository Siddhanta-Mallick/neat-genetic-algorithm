#pragma once

#include <variant>
#include <cmath>
#include <algorithm>

struct Empty
{
    double operator()(double x) const { return x; }
};

struct Sigmoid
{
    double p = 1.0;

    double operator()(double x) const { return 1.0 / (1.0 + std::exp(-x / p)); }

    bool operator==(const Sigmoid&) const = default;
};

struct ReLU
{
    double operator()(double x) const
    {
        return std::max(0.0, x);
    }

    bool operator==(const ReLU&) const = default;
};

using Activation = std::variant<Sigmoid, ReLU>;

struct ActivationFn
{
    double x;

    double operator()(const auto &fn) const
    {
        return fn(x);
    }
};
