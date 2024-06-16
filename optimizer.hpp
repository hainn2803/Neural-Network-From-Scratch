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


namespace Optimizer {

    template <typename T>
    class Basic_Optimizer {
    public:
        virtual void zero_grad() = 0;
        virtual void step() = 0;
    };

    template <typename T>
    class Gradient_Descent: public Optimizer::Basic_Optimizer<T> {
    private:
        std::vector<std::unique_ptr<Block::Layer::Linear_Layer<T>>> learnable_blocks;
        T lr;
    public:
    Gradient_Descent() {

    }
        Gradient_Descent (const std::vector<std::unique_ptr<Block::Layer::Linear_Layer<T>>>& learnable_blocks, T lr) {
            this->learnable_blocks = learnable_blocks;
            this->lr = lr;
        }

        void zero_grad() {
            for (int i = 0; i < learnable_blocks.size(); i ++) {
                learnable_blocks[i]->zero_grad();
            }
        }

       void step() {
            for (size_t i = 0; i < learnable_blocks.size(); ++i) {
                std::vector<std::valarray<T>> dW = learnable_blocks[i]->get_dW();
                std::vector<std::valarray<T>> W = learnable_blocks[i]->get_W();
                for (size_t j = 0; j < W.size(); ++j) {
                    W[j] -= this->lr * dW[j];
                }
                learnable_blocks[i]->set_W(W);

                std::valarray<T> db = learnable_blocks[i]->get_db();
                std::valarray<T> b = learnable_blocks[i]->get_b();

                b -= this->lr * db;
                learnable_blocks[i]->set_b(b);
            }
        }
    };
}