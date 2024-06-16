#include <valarray>
#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <sstream>


#include "optimizer.hpp"

namespace neural_network {

    template<typename T>
    class Neural_Network {
    private:

        std::string architecture_name;
        std::valarray<int> num_dims;
        std::vector<std::valarray<T>> layers_weight;
        std::vector<std::string> layers_name;
        std::vector<std::unique_ptr<Block::Basic_Block<T>>> layer_objects;

        Optimizer::Gradient_Descent<T> optimizer;

        std::unique_ptr<Block::Loss_Function::Cross_Entropy_Loss<T>> loss_function;

        bool _check_validity(std::vector<std::string> arch_layers) {
            // to be implementing
            return true;
        }

        std::unique_ptr<Block::Basic_Block<T>> _create_layer(const std::string& layer_name, int out_dim = 0, int inp_dim = 0) {
            if (layer_name == "linear") {
                if (out_dim <= 0 || inp_dim <= 0) {
                    throw std::invalid_argument("out_dim and inp_dim must be positive for Linear_Layer");
                }
                return std::make_unique<Block::Layer::Linear_Layer<T>>(out_dim, inp_dim);
            } else if (layer_name == "relu") {
                return std::make_unique<Block::Layer::ReLU<T>>();
            } else if (layer_name == "sigmoid") {
                return std::make_unique<Block::Layer::Sigmoid<T>>();
            } else if (layer_name == "tanh") {
                return std::make_unique<Block::Layer::Tanh<T>>();
            } else {
                throw std::invalid_argument("Unknown layer type");
            }
        }

        void _make_model() {
            int cnt = 0;
            for (const auto& x : layers_name) {
                if (x == "linear") {
                    this->layer_objects.push_back(_create_layer(x, this->num_dims[cnt+1], this->num_dims[cnt]));
                    cnt += 1;
                }
                else {
                    this->layer_objects.push_back(_create_layer(x));
                }
            }
            this->loss_function = std::make_unique<Block::Loss_Function::Cross_Entropy_Loss<T>>();
        }

    public:

        Neural_Network(std::string architecture, std::valarray<int> num_dims) {
            std::vector<std::string> arch_layers;

            std::string layer;
            std::istringstream iss(architecture);
            while (std::getline(iss, layer, '-')) {
                arch_layers.push_back(layer);
            }

            assert(_check_validity(arch_layers) && "Architecture must be valid.");
            this->layers_name = arch_layers;
            this->num_dims = num_dims;

            this->_make_model();

            // this->optimizer = std::make_unique<Optimizer::Gradient_Descent<T>>();
        }

        std::vector<std::valarray<T>> forward_logits(const std::vector<std::valarray<T>>& x_batch) {
            std::vector<std::valarray<T>> output = x_batch;
            for(int i = 0; i < layer_objects.size(); i ++) {
                output = layer_objects[i]->forward(output);
            }
            return output;
        }

        std::vector<T> predict(const std::vector<std::valarray<T>>& x_batch) {
            std::vector<std::valarray<T>> logits = this->forward_logits(x_batch);
            std::vector<T> res;
            for (int i = 0; i < logits.size(); i ++) {
                std::pair<T, std::size_t> a = find_max_and_argmax(x_batch[i]);
                std::size_t max_index = a.second;
                res.push_back(max_index);
            }
            return res;
        }

        std::pair<std::vector<std::valarray<T>>, T> forward(const std::vector<std::valarray<T>>& x_batch, const std::vector<std::valarray<T>>& target) {
            std::vector<std::valarray<T>> logits = this->forward_logits(x_batch);
            T loss = this->loss_function(logits, target);
            return std::make_pair<logits, loss>;
        }


        void backward() {
            std::vector<std::valarray<T>> dX = this->loss_function->backward();
            for(int i = layer_objects.size() - 1; i >= 0; i --) {
                dX = layer_objects[i]->backward(dX);
            }
        }

    };
}