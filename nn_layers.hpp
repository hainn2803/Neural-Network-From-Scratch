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
        virtual std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& dX) = 0;
    };


    template <typename T>
    class Linear_Layer: public Basic_Layer<T> {
    private:
        size_t inp_dim;
        size_t out_dim;
        std::vector<std::valarray<T>> W;
        std::valarray<T> b;
        std::vector<std::valarray<T>> dW;
        std::valarray<T> db;
        std::vector<std::valarray<T>> x_stored;
    public:
        Linear_Layer(size_t out_dim, size_t inp_dim) {
            this->inp_dim = inp_dim;
            this->out_dim = out_dim;
            this->W = ops_utils::init_matrix::He_initialization<T>(out_dim, inp_dim);
            this->b = ops_utils::init_matrix::generate_zeros_matrix<T>(out_dim);
            this->dW = ops_utils::init_matrix::generate_zeros_matrix<T>(out_dim, inp_dim);
            this->db = ops_utils::init_matrix::generate_zeros_matrix<T>(out_dim);
        }
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::vector<std::valarray<T>> result;
            this->x_stored = x_batch;
            for (int i = 0; i < x_batch.size(); i ++) {
                std::valarray<T> wx = ops_utils::matmul<T>(this->W, x_batch[i]);
                std::valarray<T> wxb = ops_utils::add<T>(wx, this->b);
                result.push_back(wxb);
            }
            return result;
        }

        std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& dX) override {
            std::vector<std::valarray<T>> result;
            this->dW = ops_utils::matmul(ops_utils::transpose(this->x_batch), dX);
            this->db = ops_utils::reduced_sum(dX, 0);
            std::vector<std::valarray<T>> dX_new = ops_utils::transpose(ops_utils::matmul(this->W, ops_utils::transpose(dX)));
            return dX_new;
        }

    };


    template <typename T>
    class Sigmoid: public Basic_Layer<T> {
    private:
        std::vector<std::valarray<T>> x_stored;
    public:
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::pair<size_t, size_t> x_batch_shape = ops_utils::get_shape<T>(x_batch);
            this->x_stored = x_batch;
            std::vector<std::valarray<T>> result(x_batch_shape.first, std::valarray<T>(x_batch_shape.second));

            for (int i = 0; i < x_batch_shape.first; i ++) {
                for (int j = 0; j < x_batch_shape.second; j ++) {
                    result[i][j] = act_func::forward::sigmoid_function<T>(x_batch[i][j]);
                }
            }
            return result;
        }

        std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& dX) override {
            std::pair<size_t, size_t> shape_x = ops_utils::get_shape<T>(this->x_stored);
            std::vector<std::valarray<T>> result(shape_x.first, std::valarray<T>(shape_x.second));
            for (int i = 0; i < shape_x.first; i ++) {
                for (int j = 0; j < shape_x.second; j ++) {
                    result = act_func::backward::sigmoid_function<T>(this->x_stored[i][j]);
                }
            }
            std::vector<std::valarray<T>> dX_new = ops_utils::multiply(result, dX)
            return dX_new;
        }

    };


    template <typename T>
    class ReLU: public Basic_Layer<T> {
    public:
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::pair<size_t, size_t> x_batch_shape = ops_utils::get_shape(x_batch);
            std::vector<std::valarray<T>> result(x_batch_shape.first, std::valarray<T>(x_batch_shape.second));

            for (int i = 0; i < x_batch_shape.first; i ++) {
                for (int j = 0; j < x_batch_shape.second; j ++) {
                    result[i][j] = act_func::forward::relu_function<T>(x_batch[i][j]);
                }
            }
            return result;
        }

        std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& dX) override {
            std::pair<size_t, size_t> shape_x = ops_utils::get_shape<T>(this->x_stored);
            std::vector<std::valarray<T>> result(shape_x.first, std::valarray<T>(shape_x.second));
            for (int i = 0; i < shape_x.first; i ++) {
                for (int j = 0; j < shape_x.second; j ++) {
                    result = act_func::backward::relu_function<T>(this->x_stored[i][j]);
                }
            }
            std::vector<std::valarray<T>> dX_new = ops_utils::multiply(result, dX)
            return dX_new;
        }
    };


    template <typename T>
    class Tanh: public Basic_Layer<T> {
    public:
        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
            std::pair<size_t, size_t> x_batch_shape = ops_utils::get_shape(x_batch);
            std::vector<std::valarray<T>> result(x_batch_shape.first, std::valarray<T>(x_batch_shape.second));

            for (int i = 0; i < x_batch_shape.first; i ++) {
                for (int j = 0; j < x_batch_shape.second; j ++) {
                    result[i][j] = act_func::forward::tanh_function<T>(x_batch[i][j]);
                }
            }
            return result;
        }

        std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& dX) override {
            std::pair<size_t, size_t> shape_x = ops_utils::get_shape<T>(this->x_stored);
            std::vector<std::valarray<T>> result(shape_x.first, std::valarray<T>(shape_x.second));
            for (int i = 0; i < shape_x.first; i ++) {
                for (int j = 0; j < shape_x.second; j ++) {
                    result = act_func::backward::tanh_function<T>(this->x_stored[i][j]);
                }
            }
            std::vector<std::valarray<T>> dX_new = ops_utils::multiply(result, dX)
            return dX_new;
        }
    };

}