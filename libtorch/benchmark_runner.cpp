#include <torch/torch.h>

#include <nvtx3/nvToolsExt.h>
#include "driver_types.h"
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>

#include "models.h"

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
    results_file.open("../results/libtorch_fcnet.csv", std::ios::out);
    results_file << "mnames,type,eps,loss,acc,times" << std::endl;

    // new net via reference semantics
    auto model = std::make_shared<SimpleConvNet>();
    model->to(device);

    // multi-threaded data loader for the MNIST dataset of size [batch_size, 1, 28, 28]
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../datasets/mnist-digits").map(torch::data::transforms::Stack<>()),
        /*batch_size=*/32);

    torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.01);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float lr = 0.01;

    for (size_t epoch = 1; epoch <= 15; ++epoch)
    {
        size_t batch_index = 0;
        cudaEventRecord(start);

        for (auto &batch : *data_loader)
        {
            // std::cout << batch.data.sizes() << std::endl;
            // std::cout << batch.target.sizes() << std::endl;
            
            torch::Tensor batch_data = batch.data.to(device);
            torch::Tensor batch_target = batch.target.to(device);

            optimizer.zero_grad();

            torch::Tensor prediction = model->forward(batch_data);
            torch::Tensor loss = torch::nll_loss(prediction, batch_target);
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
        results_file << "fcnet"
                     << ",training," << epoch << "," << -1 << "," << -1 << "," << milliseconds << std::endl;
    }

    results_file.close();
    return 0;
}
