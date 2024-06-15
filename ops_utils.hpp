#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <valarray> // different purpose use to vector. While vector is dynamically resizable and versatile, valarray is more numerically efficient
#include <vector>
#include <cassert>

// for testing
// #include <torch/torch.h>

namespace ops_utils {

    namespace init_matrix {

        template <typename T>
        std::vector<std::valarray<T>> generate_uniform_matrix(size_t rows, size_t cols, T min_value = -1.0, T max_value = 1.0, unsigned int seed = 28) {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<> distrib(min_value, max_value);
            std::vector<std::valarray<T>> matrix(rows, std::valarray<T>(cols));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    matrix[i][j] = distrib(gen);
                }
            }
            return matrix;
        }

        template <typename T>
        std::vector<std::valarray<T>> He_initialization(size_t rows, size_t cols, unsigned int seed = 28) {
            std::mt19937 gen(seed);
            double max_value = std::sqrt(6.0) / std::sqrt(rows + cols);
            std::uniform_real_distribution<T> distrib(-max_value, max_value);
            std::vector<std::valarray<T>> matrix(rows, std::valarray<T>(cols));
            
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    matrix[i][j] = distrib(gen);
                }
            }

            return matrix;
        }

        template <typename T>
        std::vector<std::valarray<T>> generate_zeros_matrix(size_t rows, size_t cols) {
            std::valarray<T> zero_valarray(static_cast<T>(0), cols);
            std::vector<std::valarray<T>> matrix(rows, zero_valarray);
            return matrix;
        }

        template <typename T>
        std::valarray<T> generate_zeros_matrix(size_t cols) {
            std::valarray<T> zero_valarray(static_cast<T>(0), cols);
            return zero_valarray;
        }

        template <typename T>
        std::vector<std::valarray<T>> generate_ones_matrix(size_t rows, size_t cols) {
            std::valarray<T> one_valarray(static_cast<T>(1), cols);
            std::vector<std::valarray<T>> matrix(rows, one_valarray);
            return matrix;
        }

        template <typename T>
        std::valarray<T> generate_ones_matrix(size_t cols) {
            std::valarray<T> one_valarray(static_cast<T>(1), cols);
            return one_valarray;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Extra operations for valarray
    std::valarray<double> insert_element(const std::valarray<double>& array, double value) {
        std::valarray<double> result(array.size() + 1);
        for (size_t i = 0; i < array.size(); ++i) {
            result[i] = array[i];
        }
        result[array.size()] = value;
        return result;
    }

    std::valarray<double> pop_back(const std::valarray<double>& array) {
        std::valarray<double> result(array.size() - 1);
        for (size_t i = 0; i < array.size() - 1; ++i) {
            result[i] = array[i];
        }
        return result;
    }

    std::valarray<double> pop_front(const std::valarray<double>& array) {
        std::valarray<double> result(array.size() - 1);
        for (size_t i = 1; i < array.size(); ++i) {
            result[i - 1] = array[i];
        }
        return result;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    void print_2D_matrix(const std::vector<std::valarray<T>>& A) {
        for(size_t i = 0; i < A.size(); ++i) {
            for(size_t j = 0; j < (*A.begin()).size(); ++j) {
                std::cout << A[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    template <typename T>
    void print_2D_matrix(const std::valarray<T>& A) {
        for(size_t i = 0; i < A.size(); ++i) {
            std::cout << A[i] << " ";
        }
        std::cout << std::endl;
    }
    // function for check if it is valid 2D matrix or not
    template <typename T>
    bool is_2D_matrix(const std::vector<std::valarray<T>>& A) {
        if (A.empty()) {
            return false;
        }
        size_t cols = A[0].size();
        for (const auto& row : A) {
            if (row.size() != cols) {
                return false;
            }
        }
        return true;
    }

    // function to get the shape of 2D vector
    template <typename T>
    std::pair<size_t, size_t> get_shape(const std::vector<std::valarray<T>>& A) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        return std::make_pair(A.size(), (*A.begin()).size());
    }

    // function to get the shape of 1D vector
    template <typename T>
    size_t get_shape(const std::valarray<T>& A) {
        return A.size();
    }

    template <typename T>
    std::vector<std::valarray<T>> multiply(const std::vector<std::valarray<T>>& A, const T &val) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        std::vector<std::valarray<T>> A_copied = A;
        for (auto &v : A_copied) { 
            v = v * val;
        }
        return A_copied;
    }

    template <typename T>
    std::vector<std::valarray<T>> multiply(const std::vector<std::valarray<T>>& A, const std::vector<std::valarray<T>>& B) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        assert(is_2D_matrix(B) && "Input is not a valid 2D matrix.");

        std::pair<size_t, size_t> shape_a = get_shape(A);
        std::pair<size_t, size_t> shape_b = get_shape(B);

        assert((shape_a.first == shape_b.first && shape_a.second == shape_b.second) && "Two matrices must be same size.");
        std::vector<std::valarray<T>> result = A;
        for (int i = 0; i < shape_a.first; i ++) {
            for (int j = 0; j < shape_a.second; j ++) {
                result[i][j] = A[i][j] * B[i][j];
            }
        }
        return result;
    }

    template <typename T>
    std::vector<std::valarray<T>> divide(const std::vector<std::valarray<T>>& A, const T &val) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        std::vector<std::valarray<T>> result = A;
        std::pair<size_t, size_t> shape_a = get_shape(A);
        for (auto &v : result) { 
            v = v / val;
        }
        return result;
    }

    template <typename T>
    std::vector<std::valarray<double>> divide(const std::vector<std::valarray<T>>& A, const std::vector<std::valarray<T>>& B) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        assert(is_2D_matrix(B) && "Input is not a valid 2D matrix.");

        std::pair<size_t, size_t> shape_a = get_shape(A);
        std::pair<size_t, size_t> shape_b = get_shape(B);

        assert((shape_a.first == shape_b.first && shape_a.second == shape_b.second) && "Two matrices must be same size.");
        std::vector<std::valarray<double>> result = A;
        for (int i = 0; i < shape_a.first; i ++) {
            for (int j = 0; j < shape_a.second; j ++) {
                result[i][j] = A[i][j] / B[i][j];
            }
        }
        return result;
    }

    template <typename T>
    std::vector<std::valarray<T>> add(const std::vector<std::valarray<T>>& A, const T &val) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        std::vector<std::valarray<T>> A_copied = A;
        for (auto &v: A_copied) { 
            v = v + val;
        }
        return A_copied;
    }

    template <typename T>
    std::vector<std::valarray<T>> add(const std::vector<std::valarray<T>>& A, const std::vector<std::valarray<T>>& B) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        assert(is_2D_matrix(B) && "Input is not a valid 2D matrix.");

        std::pair<size_t, size_t> shape_a = get_shape(A);
        std::pair<size_t, size_t> shape_b = get_shape(B);

        assert((shape_a.first == shape_b.first && shape_a.second == shape_b.second) && "Two matrices must be same size.");
        std::vector<std::valarray<T>> result = A;
        for (int i = 0; i < shape_a.first; i ++) {
            for (int j = 0; j < shape_a.second; j ++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        return result;
    }

    template <typename T>
    std::valarray<T> add(const std::valarray<T>& a, const std::valarray<T>& b) {
        size_t dim_a = ops_utils::get_shape(a);
        size_t dim_b = ops_utils::get_shape(b);
        assert(dim_b == dim_a && "a and b must be in same dimension.");
        std::valarray<double> result(0.0, dim_a);
        for (int i = 0; i < dim_a; i ++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    template <typename T>
    std::vector<std::valarray<T>> subtract(const std::vector<std::valarray<T>>& A, const T &val) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        std::vector<std::valarray<T>> A_copied = A;
        for (auto &v : A_copied) { 
            v = v - val;
        }
        return A_copied;
    }

    template <typename T>
    std::vector<std::valarray<T>> subtract(const T &val, const std::vector<std::valarray<T>>& A) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        std::vector<std::valarray<T>> A_copied = A;
        for (auto &v : A_copied) { 
            v = val - v;
        }
        return A_copied;
    }

    template <typename T>
    std::vector<std::valarray<T>> subtract(const std::vector<std::valarray<T>>& A, const std::vector<std::valarray<T>>& B) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        assert(is_2D_matrix(B) && "Input is not a valid 2D matrix.");

        std::pair<size_t, size_t> shape_a = get_shape(A);
        std::pair<size_t, size_t> shape_b = get_shape(B);

        assert((shape_a.first == shape_b.first && shape_a.second == shape_b.second) && "Two matrices must be same size.");
        std::vector<std::valarray<T>> result = A;
        for (int i = 0; i < shape_a.first; i ++) {
            for (int j = 0; j < shape_a.second; j ++) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
        return result;
    }

    template <typename T>
    std::valarray<T> subtract(const std::valarray<T>& a, const std::valarray<T>& b) {

        size_t dim_a = ops_utils::get_shape(a);

        size_t dim_b = ops_utils::get_shape(b);
        assert(dim_b == dim_a && "a and b must be in same dimension.");

        std::valarray<double> result(0.0, dim_a);
        for (int i = 0; i < dim_a; i ++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    template <typename T>
    std::vector<std::valarray<T>> matmul(const std::vector<std::valarray<T>>& A, const std::vector<std::valarray<T>>& B) {
        assert(is_2D_matrix(A) && "Input is not a valid 2D matrix.");
        assert(is_2D_matrix(B) && "Input is not a valid 2D matrix.");

        std::pair<size_t, size_t> shape_a = get_shape(A);
        std::pair<size_t, size_t> shape_b = get_shape(B);

        assert((shape_a.second == shape_b.first) && "Second axis of first matrix must be equal to first axis of second matrix.");

        std::vector<std::valarray<T>> result = init_matrix::generate_zeros_matrix<T>(shape_a.first, shape_b.second);

        for (int i = 0; i < shape_a.first; i ++) {
            for (int j = 0; j < shape_b.second; j ++) {
                for (int k = 0; k < shape_a.second; k ++) {
                    result[i][j] += A[i][j] * B[k][j];
                }
            }
        }
        return result;
    }

    template<typename T>
    T dot_product(const std::valarray<T>& a, const std::valarray<T>& b) {
        size_t size_a = get_shape(a);
        size_t size_b = get_shape(b);
        assert(size_a == size_b && "a and b must be in same size.");
        T result = 0.0;
        for (int i = 0; i < size_a; i ++) {
            result += (a[i] * b[i]);
        }
        return result;
    }

    template <typename T>
    std::valarray<T> matmul(const std::vector<std::valarray<T>>& W, const std::valarray<T>& x) {
        // x must have shape [D1] and W must have shape [D2, D1], matmul(W, x) must results in [D2]
        std::pair<size_t, size_t> shape_W = ops_utils::get_shape(W);
        size_t dim_x = ops_utils::get_shape(x);
        assert(shape_W.second == dim_x && "Cannot multiply W and x.");
        std::valarray<T> result(shape_W.first);
        for (int i = 0; i < shape_W.first; i ++) {
            result[i] = ops_utils::dot_product<T>(W[i], x);
        }
        return result;
    }

    template <typename T>
    std::vector<std::valarray<T>> transpose(const std::vector<std::valarray<T>>& W) {
        std::pair<size_t, size_t> shape_W = ops_utils::get_shape(W);
        std::vector<std::valarray<T>> result(shape_W.second, std::valarray<T>(shape_W.first));
        for (int i = 0; i < shape_W.first; i ++) {
            for (int j = 0; j < shape_W.second; j ++) {
                result[j][i] = W[i][j];
            }
        }
        return result;
    }

    template <typename T>
    std::valarray<T> reduced_sum(const std::vector<std::valarray<T>>& A, int dim = 0) {
        std::pair<size_t, size_t> shape_A = ops_utils::get_shape<T>(A);
        std::valarray<T> result;
        if (dim == 0) {
            result = ops_utils::init_matrix::generate_zeros_matrix<T>(shape_A.second);
        }
        else {
            result = ops_utils::init_matrix::generate_zeros_matrix<T>(shape_A.first);
        }

        for (int i = 0; i < shape_A.first; i ++) {
            for (int j = 0; j < shape_A.second; j ++) {
                if (dim == 0) {
                    result[j] += A[i][j];
                }
                else {
                    result[i] += A[i][j];
                }
            }
        }
        return result;
    }
}