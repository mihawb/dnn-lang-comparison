#include <torch/torch.h>

#include <nvtx3/nvToolsExt.h>
#include "driver_types.h"
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>

#include "src/models.h"

int main()
{
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available!" << std::endl;
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
    auto net = std::make_shared<FullyConnectedNet>();

    // multi-threaded data loader for the MNIST dataset
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../datasets/mnist-digits").map(torch::data::transforms::Stack<>()),
        /*batch_size=*/32);

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    for (size_t epoch = 1; epoch <= 15; ++epoch)
    {
        size_t batch_index = 0;
        cudaEventRecord(start);

        for (auto &batch : *data_loader)
        {
            optimizer.zero_grad();

            torch::Tensor prediction = net->forward(batch.data);
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
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
