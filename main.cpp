#include "ops_utils.hpp"
#include <iostream>

int main() {
    // std::string archi = "linear-sigmoid-linear";
    // std::valarray<int> num_dims = {2, 3, 1};
    // neural_network::Neural_Network<double> nn(archi, num_dims);
    std::vector<std::valarray<double>> X = ops_utils::init_matrix::generate_uniform_matrix<double>(3, 2);
    // ops_utils::print_2D_matrix(X);
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::vector<std::valarray<double>> y = nn.forward(X);

    // ops_utils::print_2D_matrix(y);
    std::valarray<double> x1 = ops_utils::reduced_sum<double>(X, 0);
    std::valarray<double> x2 = ops_utils::reduced_sum<double>(X, 1);

    ops_utils::print_2D_matrix(X);
    std::cout << std::endl;
    ops_utils::print_2D_matrix(x1);
    std::cout << std::endl;
    ops_utils::print_2D_matrix(x2);
    return 0;
}
