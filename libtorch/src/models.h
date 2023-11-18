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

struct Bottleneck : torch::nn::Module
{
    Bottleneck(
        int in_channels,
        int intermediate_channels,
        int expansion,
        bool is_bottleneck,
        int stride
    );

    torch::Tensor forward(torch::Tensor x);

    int in_channels;
    int intermediate_channels;
    int expansion;
    bool is_bottleneck;
    bool identity;

    torch::nn::Sequential 
        projection{nullptr},
        conv1_1x1{nullptr},
        conv2_1x1{nullptr},
        conv1_3x3{nullptr},
        conv2_3x3{nullptr};
};

// std::shared_ptr<torch::nn::Module> model_factory(std::string model_name);

#endif // _MODELS_H_