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

namespace nn_utils {
    // this namespace is for 2D case, i.e interaction of several neurons
    namespace layer {

        namespace forward {
            
            template <typename T>
            std::valarray<T> sigmoid_function(const std::valarray<T>& x) {
                std::valarray<T> result(x.size());
                for (int i = 0; i < x.size(); i ++) {
                    result[i] = act_func::forward::sigmoid_function<T>(x[i]);
                }
                return result;
            }

            template <typename T>
            std::valarray<T> relu_function(const std::valarray<T>& x) {
                std::valarray<T> result(x.size());
                for (int i = 0; i < x.size(); i ++) {
                    result[i] = act_func::forward::relu_function<T>(x[i]);
                }
                return result;
            }

            template <typename T>
            std::valarray<T> tanh_function(const std::valarray<T>& x) {
                std::valarray<T> result(x.size());
                for (int i = 0; i < x.size(); i ++) {
                    result[i] = act_func::forward::tanh_function<T>(x[i]);
                }
                return result;
            }

            template <typename T>
            std::valarray<T> linear_layer(const std::vector<std::valarray<T>>& W, const std::valarray<T>& b, const std::valarray<T>& x) {
                std::valarray<T> wx = ops_utils::matmul<T>(W, x);
                std::valarray<T> wxb = ops_utils::add<T>(wx, b);
                return wxb;
            }

        }

        namespace backward {

            template <typename T>
            std::valarray<T> sigmoid_function(const std::valarray<T>& x) {
                std::valarray<T> result(x.size());
                for (int i = 0; i < x.size(); i ++) {
                    result[i] = act_func::backward::sigmoid_function<T>(result[i]);
                }
                return result;
            }

            template <typename T>
            std::valarray<T> relu_function(const std::valarray<T>& x) {
                std::valarray<T> result(x.size());
                for (int i = 0; i < x.size(); i ++) {
                    result[i] = act_func::backward::relu_function<T>(result[i]);
                }
                return result;
            }

            template <typename T>
            std::valarray<T> tanh_function(const std::valarray<T>& x) {
                std::valarray<T> result(x.size());
                for (int i = 0; i < x.size(); i ++) {
                    result[i] = act_func::backward::tanh_function<T>(result[i]);
                }
                return result;
            }

            template <typename T>
            void linear_layer(const std::valarray<T>& dA, const std::vector<std::valarray<T>>& W, const std::valarray<T>& b, const std::valarray<T>& X) {
                std::vector<std::valarray<T>> dW = W;
                std::pair<size_t, size_t> shape_w = ops_utils::get_shape<T>(W);

            }

        }

    }

} 