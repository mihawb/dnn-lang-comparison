#ifndef _MODELS_H_
#define _MODELS_H_

#include <torch/torch.h>

struct FullyConnectedNet : torch::nn::Module
{
    FullyConnectedNet();

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif // _MODELS_H_