#include "ops_utils.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <valarray>
#include <vector>
#include <cassert>


// this namespace is just for 1D case, i.e interaction of one neuron
namespace act_func {

    namespace forward {
        template <typename T>
        T sigmoid_function(const T &x) {
            return 1.0 / (1.0 + std::exp(-x));
        }
        template <typename T>
        T relu_function(const T &x) {
            if (x >= 0) {
                return x;
            }
            else {
                return 0;
            }
        }
        template <typename T>
        T tanh_function(const T &x) {
            return (2.0 / (1.0 + std::exp(-2 * x))) - 1.0;
        }
    }

    namespace backward {

        template <typename T>
        T sigmoid_function(const T &x) {
            return x * (1 - x);
        }

        template <typename T>
        T relu_function(const T &x) {
            if (x > 0) {
                return 1;
            }
            else {
                return 0;
            }
        }

        template <typename T>
        T tanh_function(const T &x) {
            return 1 - x*x;
        }

    }

}

namespace loss_function {
    // numerically stable version
    template <typename T>
    std::valarray<T> log_softmax_function(const std::valarray<T> &x) {
        std::pair<T, std::size_t> a = ops_utils::find_max_and_argmax(x);
        T max_value_x = a.first;
        std:size_t max_index_x = a.second;
        std::valarray<T> y = x - max_value_x;
        T logsumexp = std::log(ops_utils::sum<T>(std::exp(y)));
        std::valarray<T> result = y - logsumexp;
        return result;
    }
}