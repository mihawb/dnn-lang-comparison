#include <torch/torch.h>
#include <torch/script.h>

#include <nvtx3/nvToolsExt.h>
#include "driver_types.h"
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <memory>

#include "models.h"
#include "cifar10.h"

int main()
{
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    else
    {
        std::cout << "CUDA is not available. Aborting." << std::endl;
        return -1;
    }

    std::ofstream results_file;
    results_file.open("../results/libtorch_scvnet_gpu.csv", std::ios::out);
    results_file << "mnames,type,eps,loss,acc,times" << std::endl;

    // multi-threaded data loader for the MNIST dataset of size [batch_size, 1, 28, 28]
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../datasets/mnist-digits", torch::data::datasets::MNIST::Mode::kTrain)
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/32);

    // multi-threaded data loader for the CIFAR-10 dataset of size [batch_size, 3, 32, 32]
    auto data_loader_cifar10 = torch::data::make_data_loader(
        CIFAR10{"../datasets/cifar-10-binary/cifar-10-batches-bin", CIFAR10::Mode::kTrain}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/32);

    // new net via reference semantics
    // auto model = std::make_shared<FullyConnectedNet>();
    // auto model = std::make_shared<SimpleConvNet>();
    torch::jit::script::Module model_to_be_moved = torch::jit::load("./serialized_models/resnet50_for_cifar10.pt");
    std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_to_be_moved);
    if (model == nullptr) {
        std::cout << "model load error from " << "./serialized_models/resnet50_for_cifar10.pt" << std::endl;
    }

    model->to(device);

    // workaround, since torch::jit::parameter_list is not supported by torch::optim::SGD
    std::vector<at::Tensor> model_parameters;
    for (const auto& params : model->parameters()) {
        model_parameters.push_back(params);
    }
    torch::optim::SGD optimizer(model_parameters, /*lr=*/0.01);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float lr = 0.01;
    model->train();

    for (size_t epoch = 1; epoch <= 2; ++epoch)
    {
        size_t batch_index = 0;
        cudaEventRecord(start);

        for (auto &batch : *data_loader_cifar10)
        {
            // std::cout << batch.data.sizes() << std::endl;
            // std::cout << batch.target.sizes() << std::endl;

            torch::Tensor batch_data = batch.data.to(device);
            torch::Tensor batch_target = batch.target.to(device);

            optimizer.zero_grad();

            // workaround for TorchScript models
            std::vector<torch::jit::IValue> batch_data_ivalues;
            batch_data_ivalues.push_back(batch_data);
            torch::Tensor prediction = model->forward(batch_data_ivalues).toTensor();
            // for native LibTorch models
            // torch::Tensor prediction = model->forward(batch_data);

            // for FCNet and SCVNet (last layer is log_softmax)
            // torch::Tensor loss = torch::nll_loss(prediction, batch_target);
            //for other models (last layer is softmax or logits? cant remember)
            torch::Tensor loss = torch::nll_loss(torch::log_softmax(prediction, /*dim=*/1), batch_target);

            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;

                // torch::save(net, "net.pt");
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
        results_file << "scvnet"
                     << ",training," << epoch << "," << -1 << "," << -1 << "," << milliseconds << std::endl;
    }

    results_file.close();
    return 0;
}
