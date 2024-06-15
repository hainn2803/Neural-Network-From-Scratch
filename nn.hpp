#include <valarray>
#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cassert>
#include "nn_utils.hpp"
// #include "ops_utils.hpp"

namespace neural_network {

    template<typename T>
    class Neural_Network {
    private:
        std::string architecture_name;
        std::vector<std::valarray<T>> layers_weight;
        std::vector<std::string>> layers_name;
        std::vector<std::unique_ptr<Layer::Basic_Layer<T>>> layer_objects;

        bool _check_validity(std::vector<std::string> arch_layers) {
            // to be implementing
            return true;
        }

        std::unique_ptr<Layer::Basic_Layer<T>> _create_layer(const std::string& layer_name) {
            if (layer_name == "linear") {
                return std::make_unique<Layer::Linear_Layer<T>>();
            }
            else if (layer_name == "relu") {
                return std::make_unique<Layer::ReLU<T>>();
            }
            else if (layer_name == "sigmoid") {
                return std::make_unique<Layer::Sigmoid<T>>();
            }
            else if (layer_name == "tanh") {
                return std::make_unique<Layer::Tanh<T>>();
            }
            else {
                throw std::invalid_argument("Unknown layer type");
            }
        }

        void _make_model() {
            for (const auto& x : layers_name) {
                layer_objects.push_back(_create_layer(x));
            }
        }

    public:
        Neural_Network(std::string architecture) {
            std::vector<std::string> arch_layers;

            std::string layer;
            std::istringstream iss(architecture);
            while (std::getline(iss, layer, '-')) {
                arch_layers.push_back(layer);
            }

            assert(_check_validity(arch_layers) && "Architecture must be valid.");
            layers_name = arch_layers;

            this->_make_model();
        }


    };
}