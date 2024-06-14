#include "ops_utils.hpp"
#include "data_utils.hpp"
#include <iostream>

int main() {
    // std::string dataset_name = "dataset/iris.csv";
    // auto dataset = get_data(dataset_name);
    // auto X = dataset.first;
    // auto Y = dataset.second;

    // for (auto &x: X) {
    //     std::pair<size_t, size_t> mat_shape = ops_utils::get_shape(x);
    //     std::cout << "Shape of random_matrix: (" << mat_shape.first << ", " << mat_shape.second << ")" << std::endl;
    // }

    size_t rows_a = 3;
    size_t cols_a = 4;
    size_t rows_b = 4;
    size_t cols_b = 5;
    double min_value = 0.0;
    double max_value = 10.0;
    double scalar = 2.0;

    // Generate random matrix
    auto A = ops_utils::init_matrix::generate_uniform_matrix(rows_a, cols_a, min_value, max_value);
    auto B = ops_utils::init_matrix::generate_uniform_matrix(rows_b, cols_b, min_value, max_value);

    auto C = ops_utils::matmul(A, B);

    std::pair<size_t, size_t> mat_shape = ops_utils::get_shape(C);
    std::cout << "Shape of random_matrix: (" << mat_shape.first << ", " << mat_shape.second << ")" << std::endl;

    return 0;
}
