#ifndef _MODELS_H_
#define _MODELS_H_

#include <torch/torch.h>

struct FullyConnectedNet : torch::nn::Module
{
    FullyConnectedNet(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

struct SimpleConvNet : torch::nn::Module
{
    SimpleConvNet(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear dense{nullptr}, classifier{nullptr};
};

#endif // _MODELS_H_