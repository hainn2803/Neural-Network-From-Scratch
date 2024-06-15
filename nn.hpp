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


#include "nn_layers.hpp"

namespace neural_network {

    template<typename T>
    class Neural_Network {
    public:

        std::string architecture_name;
        std::valarray<int> num_dims;
        std::vector<std::valarray<T>> layers_weight;
        std::vector<std::string> layers_name;
        std::vector<std::unique_ptr<Layer::Basic_Layer<T>>> layer_objects;

        bool _check_validity(std::vector<std::string> arch_layers) {
            // to be implementing
            return true;
        }

        std::unique_ptr<Layer::Basic_Layer<T>> _create_layer(const std::string& layer_name, int out_dim = 0, int inp_dim = 0) {
            if (layer_name == "linear") {
                if (out_dim <= 0 || inp_dim <= 0) {
                    throw std::invalid_argument("out_dim and inp_dim must be positive for Linear_Layer");
                }
                return std::make_unique<Layer::Linear_Layer<T>>(out_dim, inp_dim);
            } else if (layer_name == "relu") {
                return std::make_unique<Layer::ReLU<T>>();
            } else if (layer_name == "sigmoid") {
                return std::make_unique<Layer::Sigmoid<T>>();
            } else if (layer_name == "tanh") {
                return std::make_unique<Layer::Tanh<T>>();
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
        }

        std::vector<std::valarray<T>> forward(const std::vector<std::valarray<T>>& x_batch) {
            std::vector<std::valarray<T>> output = x_batch;
            for(int i = 0; i < layer_objects.size(); i ++) {
                output = layer_objects[i]->forward(output);
            }
            return output;
        }


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
        }


    };
}