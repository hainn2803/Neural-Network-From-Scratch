#include "nn.hpp"
#include <iostream>

int main() {
    std::string archi = "linear-sigmoid-linear";
    std::valarray<int> num_dims = {2, 3, 1};
    neural_network::Neural_Network<double> nn(archi, num_dims);
    std::vector<std::valarray<double>> X = ops_utils::init_matrix::generate_uniform_matrix<double>(3, 2);
    ops_utils::print_2D_matrix(X);
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < nn.layer_objects.size(); i ++) {
        if (nn.layers_name[i] == "linear") {
            auto* linear_layer = dynamic_cast<Layer::Linear_Layer<double>*>(nn.layer_objects[i].get());
            std::cout << i << std::endl;
            ops_utils::print_2D_matrix(linear_layer->W);
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
    std::vector<std::valarray<double>> y = nn.forward(X);

    ops_utils::print_2D_matrix(y);
    return 0;
}
