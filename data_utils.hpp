#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <valarray>
#include <string>
#include <utility>
#include <cstdlib>

#include "ops_utils.hpp"

std::vector<std::vector<std::valarray<double>>> minmax_scaler(const std::vector<std::vector<std::valarray<double>>>& X, double min_value, double max_value) {
    // Implement your min-max scaling logic here
    // For simplicity, return X as is in this placeholder
    return X;
}

std::pair<std::vector<std::vector<std::valarray<double>>>, std::vector<std::vector<std::valarray<double>>>> get_data(const std::string& file_name, const bool& last_label = true, const bool& normalize = true, const int& skip_lines = 1) {
    std::ifstream in_file;
    in_file.open(file_name.c_str(), std::ios::in);
    // If there is any problem in opening file
    if (!in_file.is_open()) {
        std::cerr << "ERROR (" << __func__ << ") : ";
        std::cerr << "Unable to open file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::vector<std::valarray<double>>> X, Y;  // To store X and Y
    std::string line;  // To store each line

    // Skip lines
    for (int i = 0; i < skip_lines; i++) {
        std::getline(in_file, line, '\n');  // Ignore line
    }

    // While file has information
    while (!in_file.eof() && std::getline(in_file, line, '\n')) {
        std::valarray<double> x_data, y_data;  // To store single sample and label
        std::stringstream ss(line);  // Constructing stringstream from line
        std::string token;  // To store each token in line (separated by ',')

        while (std::getline(ss, token, ',')) {  // For each token
            // Insert numerical value of token in x_data
            x_data = ops_utils::insert_element(x_data, std::stod(token));
        }

        // If label is in last column
        if (last_label) {
            y_data.resize(1);  // Assuming 1 neuron for simplicity
            y_data[0] = x_data[x_data.size() - 1];
            x_data = ops_utils::pop_back(x_data);  // Remove label from x_data
        } else {
            y_data.resize(1);  // Assuming 1 neuron for simplicity
            y_data[0] = x_data[0];
            x_data = ops_utils::pop_front(x_data);  // Remove label from x_data
        }

        // Push collected X_data and y_data in X and Y
        X.push_back({x_data});
        Y.push_back({y_data});
    }
    if (normalize) {
        X = minmax_scaler(X, 0.01, 1.0);
    }
    in_file.close();  // Closing file
    return std::make_pair(X, Y);  // Return pair of X and Y
}

#endif