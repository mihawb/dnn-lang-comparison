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

struct ExtendedConvNet : torch::nn::Module
{
    ExtendedConvNet(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential conv1{nullptr}, conv2{nullptr}, 
    extended_convs{nullptr}, extended_dense{nullptr};
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

struct ResNet50 : torch::nn::Module
{
    ResNet50(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential
        process_input{nullptr},
        block1{nullptr},
        block2{nullptr},
        block3{nullptr},
        block4{nullptr};
    torch::nn::AdaptiveAvgPool2d average_pool{nullptr};
    torch::nn::Linear fc1{nullptr};
};

struct Generator : torch::nn::Module
{
    Generator(int n_channels = 3, int latent_vec_size = 100, int feat_map_size = 3);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential main{nullptr};
};

struct Discriminator : torch::nn::Module
{
    Discriminator(int n_channels = 3, int feat_map_size = 3);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential main{nullptr};
};

struct ResBlock : torch::nn::Module
{
    ResBlock(int in_channels, int out_channels);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential base1{nullptr}, base2{nullptr};
    torch::nn::MaxPool2d mpool{nullptr};
};

struct SODNet : torch::nn::Module
{
    SODNet(int in_channels, int first_output_channels);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential main{nullptr};
};

#endif // _MODELS_H_