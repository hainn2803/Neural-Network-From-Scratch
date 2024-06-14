cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(OptUtils)

# Set the C++ standard to C++14 or higher
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Specify the path to your libtorch installation
set(Torch_DIR "/home/ubuntu/learn_cuda/libtorch/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)

# Add the executable
add_executable(opt_utils /home/ubuntu/learn_cuda/neural_network/ops_utils.cpp)

# Link the Torch library
target_link_libraries(opt_utils "${TORCH_LIBRARIES}")

# Add include directories
target_include_directories(opt_utils PUBLIC "/home/ubuntu/learn_cuda/libtorch/include" "/home/ubuntu/learn_cuda/libtorch/include/torch/csrc/api/include")

# Ensure rpath includes the libtorch library path
set(CMAKE_BUILD_RPATH "/home/ubuntu/learn_cuda/libtorch/lib")