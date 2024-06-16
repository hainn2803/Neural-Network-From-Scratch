#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <valarray>
#include <vector>
#include <cassert>

#include "nn_utils.hpp"

namespace Block {

    template <typename T>
    class Basic_Block {
    public:
        virtual std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) = 0;
        virtual std::vector<std::valarray<T>> backward(const std::vector<std::valarray<T>>& dX) = 0;
    };

    namespace Layer {

        template <typename T>
        class Linear_Layer: public Block::Basic_Block<T> {
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
                this->dW = ops_utils::matmul(ops_utils::transpose(this->x_stored), dX);
                this->db = ops_utils::reduced_sum(dX, 0);
                std::vector<std::valarray<T>> dX_new = ops_utils::transpose(ops_utils::matmul(this->W, ops_utils::transpose(dX)));
                return dX_new;
            }

            void zero_grad() {
                this->dW = ops_utils::init_matrix::generate_zeros_matrix<T>(this->out_dim, this->inp_dim);
                this->db = ops_utils::init_matrix::generate_zeros_matrix<T>(this->out_dim);
            }

            std::vector<std::valarray<T>>& get_W() {
                return W;
            }
            void set_W(const std::vector<std::valarray<T>>& new_W) {
                W = new_W;
            }
            std::valarray<T>& get_b() {
                return b;
            }
            void set_b(const std::valarray<T>& new_b) {
                b = new_b;
            }
            std::vector<std::valarray<T>>& get_dW() {
                return dW;
            }
            std::valarray<T>& get_db() {
                return db;
            }

        };


        template <typename T>
        class Sigmoid: public Block::Basic_Block<T> {
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
                        result[i][j] = act_func::backward::sigmoid_function<T>(this->x_stored[i][j]);
                    }
                }
                std::vector<std::valarray<T>> dX_new = ops_utils::multiply(result, dX);
                return dX_new;
            }

        };


        template <typename T>
        class ReLU: public Block::Basic_Block<T> {
        private:
            std::vector<std::valarray<T>> x_stored;
        public:
            std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
                this->x_stored = x_batch;
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
                        result[i][j] = act_func::backward::relu_function<T>(this->x_stored[i][j]);
                    }
                }
                std::vector<std::valarray<T>> dX_new = ops_utils::multiply(result, dX);
                return dX_new;
            }
        };


        template <typename T>
        class Tanh: public Block::Basic_Block<T> {
        private:
            std::vector<std::valarray<T>> x_stored;
        public:
            std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) override {
                this->x_stored = x_batch;
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
                        result[i][j] = act_func::backward::tanh_function<T>(this->x_stored[i][j]);
                    }
                }
                std::vector<std::valarray<T>> dX_new = ops_utils::multiply(result, dX);
                return dX_new;
            }
        };

    }

    namespace Loss_Function {

        template <typename T>
        class Binary_Cross_Entropy_Loss {
        public:
            std::valarray<T> forward(const std::vector<std::valarray<T>>& pred, const std::vector<std::valarray<T>>& target) {

            }
        };

        template <typename T>
        class Cross_Entropy_Loss {
        private:
            std::string reduction;
            std::vector<std::valarray<T>> logits;
            std::vector<std::valarray<T>> target;
        public:
            Cross_Entropy_Loss() {
                this->reduction = "mean";
            }
            Cross_Entropy_Loss(const std::string& reduction) {
                this->reduction = reduction;
            }
            T forward(const std::vector<std::valarray<T>>& pred, const std::vector<std::valarray<T>>& target) {
                this->logits = pred;
                this->target = target;
                std::pair<size_t, size_t> shape_pred = ops_utils::get_shape<T>(pred);
                std::pair<size_t, size_t> shape_target = ops_utils::get_shape<T>(target);
                assert((shape_pred.first == shape_target.first && shape_pred.second == shape_target.second) && "prediction and target must be in same size.");
                T res = 0;
                for (int i = 0; i < pred.size(); i ++){
                    T abc = ops_utils::sum(loss_function::log_softmax_function<T>(pred) * target[i]);
                    res += abc;
                }
                if (this->reduction == "mean") {
                    return res / shape_pred.first;
                }
                else {
                    return res;
                }
            }

            std::vector<std::valarray<T>> backward() {

                std::pair<size_t, size_t> shape_pred = ops_utils::get_shape<T>(this->logits);
                std::pair<size_t, size_t> shape_target = ops_utils::get_shape<T>(this->target);
                assert((shape_pred.first == shape_target.first && shape_pred.second == shape_target.second) && "prediction and target must be in same size.");

                std::vector<std::valarray<T>> minus_pred = ops_utils::subtract<T>(1, this->logits);
                std::vector<std::valarray<T>> minus_target = ops_utils::subtract<T>(1, this->target);

                std::vector<std::valarray<T>> t_divide_o = ops_utils::divide<T>(this->target, this->logits);
                std::vector<std::valarray<T>> minus_t_divide_o = ops_utils::divide<T>(minus_target, minus_pred);

                std::vector<std::valarray<T>> res = ops_utils::subtract<T>(minus_t_divide_o, t_divide_o);

                return res;
            }
        };
    }

}