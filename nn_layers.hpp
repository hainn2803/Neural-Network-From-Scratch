#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <valarray>
#include <vector>
#include <cassert>

#include "nn_utils.hpp"

namespace Layer {

    template <typename T>
    class Basic_Layer {
    public:
        virtual std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) = 0;
        // virtual std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& x_batch) = 0;
    };

    template <typename T>
    class Linear_Layer: public Basic_Layer<T> {
    public:
        size_t inp_dim;
        size_t out_dim;
        std::vector<std::valarray<T>> W;
        std::valarray<T> b;
    public:
        Linear_Layer(size_t out_dim, size_t inp_dim) {
            this->inp_dim = inp_dim;
            this->out_dim = out_dim;
            this->W = ops_utils::init_matrix::He_initialization<T>(out_dim, inp_dim);
            this->b = ops_utils::init_matrix::generate_zeros_matrix<T>(out_dim);
        }

        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::vector<std::valarray<T>> result;
            for (int i = 0; i < x_batch.size(); i ++) {
                std::valarray<T> y = nn_utils::layer::forward::linear_layer<T>(W, b, x_batch[i]);
                result.push_back(y);
            }
            return result;
        }
    };

    template <typename T>
    class Sigmoid: public Basic_Layer<T> {
    public:
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::vector<std::valarray<T>> result;
            for (int i = 0; i < x_batch.size(); i ++) {
                std::valarray<T> y = nn_utils::layer::forward::sigmoid_function<T>(x_batch[i]);
                result.push_back(y);
            }
            return result;
        }
    };

    template <typename T>
    class ReLU: public Basic_Layer<T> {
    public:
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::vector<std::valarray<T>> result;
            for (int i = 0; i < x_batch.size(); i ++) {
                std::valarray<T> y = nn_utils::layer::forward::relu_function<T>(x_batch[i]);
                result.push_back(y);
            }
            return result;
        }
    };

    template <typename T>
    class Tanh: public Basic_Layer<T> {
    public:
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::vector<std::valarray<T>> result;
            for (int i = 0; i < x_batch.size(); i ++) {
                std::valarray<T> y = nn_utils::layer::forward::tanh_function<T>(x_batch[i]);
                result.push_back(y);
            }
            return result;
        }
    };

}