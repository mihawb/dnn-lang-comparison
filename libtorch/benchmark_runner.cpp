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

int main() {
    // testy xdd
    torch::Tensor x = torch::randn({1,64,112,112});
    auto model_to_be_moved = Bottleneck(64,64,4,true,2);
    auto model = std::make_shared<Bottleneck>(model_to_be_moved);
    torch::Tensor out = model->forward(x);
    std::cout << torch::_shape_as_tensor(out) << std::endl;
}

int main_main()
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
    results_file.open("../results/libtorch.csv", std::ios::out);
    results_file << "model_name,type,epoch,loss,performance,elapsed_time" << std::endl;

    int batch_size = 96;
    int test_batch_size = 128;
    int epochs = 5;
    float lr = 0.02;
    float momentum = 0.9;
    int num_classes = 10;
    int log_interval = 200;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    

    // multi-threaded data loader for the MNIST dataset of size [batch_size, 1, 28, 28]
    auto train_dl_mnist = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../datasets/mnist-digits", torch::data::datasets::MNIST::Mode::kTrain)
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/batch_size);

    auto test_dl_mnist = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../datasets/mnist-digits", torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/test_batch_size);

    // multi-threaded data loader for the CIFAR-10 dataset of size [batch_size, 3, 32, 32]
    auto train_dl_cifar10 = torch::data::make_data_loader(
        CIFAR10{"../datasets/cifar-10-binary/cifar-10-batches-bin", CIFAR10::Mode::kTrain}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/batch_size);

    auto test_dl_cifar10 = torch::data::make_data_loader(
        CIFAR10{"../datasets/cifar-10-binary/cifar-10-batches-bin", CIFAR10::Mode::kTest}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/test_batch_size);

    // ==================================================================FullyConnectedNet
    // new net via reference semantics
    auto model_fcnet = std::make_shared<FullyConnectedNet>();
    model_fcnet->to(device);
    torch::optim::SGD optimizer_fcnet(model_fcnet->parameters(), /*lr=*/0.01);

    model_fcnet->train();
    for (size_t epoch = 1; epoch <= epochs; ++epoch)
    {
        size_t batch_index = 0;
        double running_loss = 0.0;
        int running_corrects = 0;
        int num_samples = 0;
        cudaEventRecord(start);

        for (auto &batch : *train_dl_mnist)
        {
            num_samples += batch.data.size(0);
            torch::Tensor batch_data = batch.data.to(device);
            torch::Tensor batch_target = batch.target.to(device);

            optimizer_fcnet.zero_grad();

            torch::Tensor outputs = model_fcnet->forward(batch_data);
            torch::Tensor loss = torch::nll_loss(outputs, batch_target);
            
            loss.backward();
            optimizer_fcnet.step();
            running_loss += loss.item<double>();
            // choosing class with max logit (outputs is a vector of vectors of class probabilities)
            torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
            running_corrects += torch::sum(predictions == batch_target).item<int>();

            if (++batch_index % log_interval == 0)
            {
                std::cout << "[" << epoch << "]\t[" << batch_index <<
                    "]\tLoss: " << loss.item<float>() << std::endl;
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);


        std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
        results_file << "FullyConnectedNet,training," 
            << epoch << "," << running_loss / (batch_index+1) << ","
            << (float)running_corrects / (float)num_samples << ","
            << milliseconds << std::endl;
    }

    model_fcnet->eval();
    for (auto &batch : *test_dl_mnist) {
        cudaEventRecord(start);

        torch::Tensor batch_data = batch.data.to(device);
        torch::Tensor batch_target = batch.target.to(device);

        torch::Tensor outputs = model_fcnet->forward(batch_data);
        torch::Tensor loss = torch::nll_loss(outputs, batch_target);
        torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
        int corrects = torch::sum(predictions == batch_target).item<int>();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
        results_file << "FullyConnectedNet,inference," 
            << 1 << "," << loss.item<float>() << ","
            << (float)corrects / (float)batch.data.size(0) << ","
            << milliseconds << std::endl;
        break;
    }

    // ======================================================================SimpleConvNet
    auto model_scvnet = std::make_shared<SimpleConvNet>();
    model_scvnet->to(device);
    torch::optim::SGD optimizer_scvnet(model_scvnet->parameters(), /*lr=*/0.01);

    model_scvnet->train();
    for (size_t epoch = 1; epoch <= epochs; ++epoch)
    {
        size_t batch_index = 0;
        double running_loss = 0.0;
        int running_corrects = 0;
        int num_samples = 0;
        cudaEventRecord(start);

        for (auto &batch : *train_dl_mnist)
        {
            num_samples += batch.data.size(0);
            torch::Tensor batch_data = batch.data.to(device);
            torch::Tensor batch_target = batch.target.to(device);

            optimizer_scvnet.zero_grad();

            torch::Tensor outputs = model_scvnet->forward(batch_data);
            torch::Tensor loss = torch::nll_loss(outputs, batch_target);
            
            loss.backward();
            optimizer_scvnet.step();
            running_loss += loss.item<double>();
            torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
            running_corrects += torch::sum(predictions == batch_target).item<int>();

            if (++batch_index % log_interval == 0)
            {
                std::cout << "[" << epoch << "]\t[" << batch_index <<
                    "]\tLoss: " << loss.item<float>() << std::endl;
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);


        std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
        results_file << "SimpleConvNet,training," 
            << epoch << "," << running_loss / (batch_index+1) << ","
            << (float)running_corrects / (float)num_samples << ","
            << milliseconds << std::endl;
    }

    model_scvnet->eval();
    for (auto &batch : *test_dl_mnist) {
        cudaEventRecord(start);

        torch::Tensor batch_data = batch.data.to(device);
        torch::Tensor batch_target = batch.target.to(device);

        torch::Tensor outputs = model_scvnet->forward(batch_data);
        torch::Tensor loss = torch::nll_loss(outputs, batch_target);
        torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
        int corrects = torch::sum(predictions == batch_target).item<int>();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        results_file << "SimpleConvNet,inference," 
            << 1 << "," << loss.item<float>() << ","
            << (float)corrects / (float)batch.data.size(0) << ","
            << milliseconds << std::endl;
        break;
    }

    results_file.close();
    return 0;
}