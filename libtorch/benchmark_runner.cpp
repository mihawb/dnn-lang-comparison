#include <torch/torch.h>
#include <torch/script.h>

#include <nvtx3/nvToolsExt.h>
#include "driver_types.h"
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <chrono>

#include "models.h"
#include "cifar10.h"
#include "celeba.h"
#include "adam.h"

std::string models[] = {"ResNet-50", "DenseNet-121", "MobileNet-v2", "ConvNeXt-Tiny"};
bool RUN_FCNET          = false;
bool RUN_SCVNET         = true;
bool RUN_ECVNET         = false;
bool RUN_NATIVE_RESNET  = false;
bool RUN_TORCHSCRIPT    = false;
bool RUN_DCGAN          = false;
bool RUN_SODNET         = false;

int main_scvnet_dtype_test()
{
    auto mnist = torch::data::datasets::MNIST(
        "../datasets/mnist-digits",
        torch::data::datasets::MNIST::Mode::kTrain);
    torch::Tensor example = mnist.get(0).data;
    std::cout << example.dtype() << std::endl;

    torch::Tensor ones = torch::ones_like(example, torch::TensorOptions().dtype(torch::kFloat16));
    std::cout << "kF16 dtype repr: " << ones.dtype() << std::endl;
    ones = torch::ones_like(example, torch::TensorOptions().dtype(torch::kFloat32));
    std::cout << "kF32 dtype repr: " << ones.dtype() << std::endl;
    ones = torch::ones_like(example, torch::TensorOptions().dtype(torch::kFloat64));
    std::cout << "kF64 dtype repr: " << ones.dtype() << std::endl;

    auto model_scvnet = std::make_shared<SimpleConvNet>();
    std::cout << model_scvnet->conv1->modules().at(0)->parameters().at(0).dtype() << std::endl;
    
    return 0;
}

int main_adam_test()
{
    std::string root = "../datasets/ADAM/Training1200";
    std::pair<torch::Tensor, torch::Tensor> data = read_data(root);
    std::cout << data.first.sizes() << " " << data.second.sizes() << std::endl;
    std::cout << data.second[0] << std::endl;
    std::cout << data.first[0][0][100][100] << std::endl;

    auto train_dl_adam = torch::data::make_data_loader(
        ADAM{"../datasets/ADAM/Training1200", ADAM::Mode::kTrain}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/8);

    auto test_dl_adam = torch::data::make_data_loader(
        ADAM{"../datasets/ADAM/Training1200", ADAM::Mode::kTest}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/16);

    bool first = true;
    int num_train = 0;
    int num_test = 0;

    for (auto &batch : *train_dl_adam)
    {
        if (first)
        {
            first = false;
            std::cout << "batch data shape: " << batch.data.sizes() << std::endl;
            std::cout << "batch target shape: " << batch.target.sizes() << std::endl;
        }
        num_train += batch.data.size(0);
    }
    for (auto &batch : *test_dl_adam)
        num_test += batch.data.size(0);

    int num_total = num_train + num_test;
    std::cout << "sizes: (train/test/total): "
              << num_train << "/"
              << num_test << "/"
              << num_total << std::endl;

    return 0;
}

int main_sodnet_test()
{
    torch::Tensor x = torch::zeros({2, 3, 256, 256});

    auto model_resblock = std::make_shared<ResBlock>(ResBlock(3, 16));
    torch::Tensor y = model_resblock->forward(x);
    std::cout << "resblock output size: " << y.sizes() << std::endl; // [2, 16, 256, 256] which is correct

    auto model_sodnet = std::make_shared<SODNet>(SODNet(3, 16));
    torch::Tensor z = model_sodnet->forward(x);
    std::cout << "sodnet output size: " << z.sizes() << " and content: " << z << std::endl;

    return 0;
}

int main_batch_loading_test()
{
    int batch_size = 96;
    auto celeba = CELEBA{"../datasets/celeba_test", batch_size};
    torch::Tensor first_batch = celeba.get_batch_by_id(0);
    std::cout << "celeba first batch sizes: " << first_batch.sizes() << std::endl;

    torch::Tensor last_batch = celeba.get_batch_by_id(132); // id already to high, 131 is max
    std::cout << "celeba last (" << 131 << ") batch sizes: " << last_batch.sizes() << std::endl;

    return 0;
}

int main()
{
    //===================================================================environment setup

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

    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(4);

    //======================================training configuration and shared declarations

    std::ofstream results_file;
    results_file.open("../results/libtorch.csv", std::ios::out);
    results_file << "model_name,phase,epoch,loss,performance,elapsed_time" << std::endl;

    int batch_size = 96;
    int test_batch_size = 128;
    int epochs = 12;
    float lr = 0.01;
    float momentum = 0.9;
    int num_classes = 10;
    int log_interval = 100;
    int latency_warmup_steps = 1000;

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

    // multi-threaded data loader for the ADAM dataset of size [batch_size, 3, 256, 256]
    auto train_dl_adam = torch::data::make_data_loader(
        ADAM{"../datasets/ADAM/Training1200", ADAM::Mode::kTrain}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/8);

    auto test_dl_adam = torch::data::make_data_loader(
        ADAM{"../datasets/ADAM/Training1200", ADAM::Mode::kTest}
            .map(torch::data::transforms::Stack<>()),
        /*batch_size=*/16);

    if (RUN_FCNET) //====================================================FullyConnectedNet
    {
        std::cout << "Benchmarks for FullyConnectedNet begin." << std::endl;
        // new net via reference semantics
        auto model_fcnet = std::make_shared<FullyConnectedNet>();
        model_fcnet->to(device);
        torch::optim::SGD optimizer_fcnet(model_fcnet->parameters(), /*lr=*/lr);

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
                    std::cout << "[" << epoch << "]\t[" << batch_index << "]\tLoss: " << loss.item<float>() << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << "FullyConnectedNet,training,"
                         << epoch << "," << running_loss / (batch_index + 1) << ","
                         << (float)running_corrects / (float)num_samples << ","
                         << milliseconds << std::endl;
        }

        model_fcnet->eval();
        int batch_index = 0;
        double running_loss = 0.0;
        int corrects = 0;
        int num_samples = 0;

        {
            torch::NoGradGuard no_grad;

            cudaEventRecord(start);
            for (auto &batch : *test_dl_mnist)
            {
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                torch::Tensor outputs = model_fcnet->forward(batch_data);
                torch::Tensor loss = torch::nll_loss(outputs, batch_target);
                torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));

                batch_index++;
                corrects += torch::sum(predictions == batch_target).item<int>();
                num_samples += batch.data.size(0);
                running_loss += loss.item<double>();
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
        results_file << "FullyConnectedNet,inference,"
                     << 1 << "," << running_loss / (double)batch_index << ","
                     << (float)corrects / (float)num_samples << ","
                     << milliseconds << std::endl;

        {
            torch::NoGradGuard no_grad;

            int warmup = -latency_warmup_steps;
            bool warmup_finished = false;

            for (auto &batch : *test_dl_mnist)
            {
                warmup_finished = warmup_finished || warmup > latency_warmup_steps;
                warmup += batch_size;

                torch::Tensor batch_data = batch.data.to(device);

                for (int i = 0; i < (i <= warmup_finished ? batch_size : epochs); i++)
                {
                    torch::Tensor sample = batch_data[i];
                    sample = sample.unsqueeze(0);

                    cudaEventRecord(start);
                    torch::Tensor outputs = model_fcnet->forward(batch_data);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    if (warmup_finished)
                    {
                        std::cout << "[Epoch " << i << "] Latency: "
                                  << milliseconds << " ms" << std::endl;
                        results_file << "FullyConnectedNet,latency," << i + 1 << ",-1,-1,"
                                     << milliseconds << std::endl;
                    }
                }

                if (warmup_finished)
                    break;
            }
        }
    }

    if (RUN_SCVNET) //=======================================================SimpleConvNet
    {
        std::cout << "Benchmarks for SimpleConvNet begin." << std::endl;
        auto model_scvnet = std::make_shared<SimpleConvNet>();
        model_scvnet->to(device);
        torch::optim::SGD optimizer_scvnet(model_scvnet->parameters(), /*lr=*/lr);

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
                    std::cout << "[" << epoch << "]\t[" << batch_index << "]\tLoss: " << loss.item<float>() << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << "SimpleConvNet,training,"
                         << epoch << "," << running_loss / (batch_index + 1) << ","
                         << (float)running_corrects / (float)num_samples << ","
                         << milliseconds << std::endl;
        }

        model_scvnet->eval();
        int batch_index = 0;
        double running_loss = 0.0;
        int corrects = 0;
        int num_samples = 0;

        {
            torch::NoGradGuard no_grad;

            cudaEventRecord(start);
            for (auto &batch : *test_dl_mnist)
            {
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                torch::Tensor outputs = model_scvnet->forward(batch_data);
                torch::Tensor loss = torch::nll_loss(outputs, batch_target);
                torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
                int corrects = torch::sum(predictions == batch_target).item<int>();

                batch_index++;
                corrects += torch::sum(predictions == batch_target).item<int>();
                num_samples += batch.data.size(0);
                running_loss += loss.item<double>();
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
        results_file << "SimpleConvNet,inference,"
                     << 1 << "," << running_loss / (double)batch_index << ","
                     << (float)corrects / (float)num_samples << ","
                     << milliseconds << std::endl;

        {
            torch::NoGradGuard no_grad;

            int warmup = -latency_warmup_steps;
            bool warmup_finished = false;

            for (auto &batch : *test_dl_mnist)
            {
                warmup_finished = warmup_finished || warmup > latency_warmup_steps;
                warmup += batch_size;

                torch::Tensor batch_data = batch.data.to(device);

                for (int i = 0; i < (i <= warmup_finished ? batch_size : epochs); i++)
                {
                    torch::Tensor sample = batch_data[i];
                    sample = sample.unsqueeze(0);

                    cudaEventRecord(start);
                    torch::Tensor outputs = model_scvnet->forward(batch_data);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    if (warmup_finished)
                    {
                        std::cout << "[Epoch " << i << "] Latency: "
                                  << milliseconds << " ms" << std::endl;
                        results_file << "SimpleConvNet,latency," << i + 1 << ",-1,-1,"
                                     << milliseconds << std::endl;
                    }
                }

                if (warmup_finished)
                    break;
            }
        }
    }

    if (RUN_ECVNET) //=====================================================ExtendedConvNet
    {
        std::cout << "Benchmarks for ExtendedConvNet begin." << std::endl;
        auto model_scvnet = std::make_shared<SimpleConvNet>();
        model_scvnet->to(device);
        torch::optim::SGD optimizer_scvnet(model_scvnet->parameters(), /*lr=*/lr);

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
                    std::cout << "[" << epoch << "]\t[" << batch_index << "]\tLoss: " << loss.item<float>() << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << "ExtendedConvNet,training,"
                         << epoch << "," << running_loss / (batch_index + 1) << ","
                         << (float)running_corrects / (float)num_samples << ","
                         << milliseconds << std::endl;
        }

        model_scvnet->eval();
        int batch_index = 0;
        double running_loss = 0.0;
        int corrects = 0;
        int num_samples = 0;

        {
            torch::NoGradGuard no_grad;

            cudaEventRecord(start);
            for (auto &batch : *test_dl_mnist)
            {
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                torch::Tensor outputs = model_scvnet->forward(batch_data);
                torch::Tensor loss = torch::nll_loss(outputs, batch_target);
                torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
                int corrects = torch::sum(predictions == batch_target).item<int>();

                batch_index++;
                corrects += torch::sum(predictions == batch_target).item<int>();
                num_samples += batch.data.size(0);
                running_loss += loss.item<double>();
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
        results_file << "ExtendedConvNet,inference,"
                     << 1 << "," << running_loss / (double)batch_index << ","
                     << (float)corrects / (float)num_samples << ","
                     << milliseconds << std::endl;

        {
            torch::NoGradGuard no_grad;

            int warmup = -latency_warmup_steps;
            bool warmup_finished = false;

            for (auto &batch : *test_dl_mnist)
            {
                warmup_finished = warmup_finished || warmup > latency_warmup_steps;
                warmup += batch_size;

                torch::Tensor batch_data = batch.data.to(device);

                for (int i = 0; i < (i <= warmup_finished ? batch_size : epochs); i++)
                {
                    torch::Tensor sample = batch_data[i];
                    sample = sample.unsqueeze(0);

                    cudaEventRecord(start);
                    torch::Tensor outputs = model_scvnet->forward(batch_data);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    if (warmup_finished)
                    {
                        std::cout << "[Epoch " << i << "] Latency: "
                                  << milliseconds << " ms" << std::endl;
                        results_file << "ExtendedConvNet,latency," << i + 1 << ",-1,-1,"
                                     << milliseconds << std::endl;
                    }
                }

                if (warmup_finished)
                    break;
            }
        }
    }

    if (RUN_NATIVE_RESNET) //====================================================ResNet-50
    {
        std::cout << "Benchmarks for native ResNet-50 begin." << std::endl;
        auto model_resnet50_native = std::make_shared<ResNet50>();
        model_resnet50_native->to(device);
        torch::optim::SGD optimizer_resnet50_native(model_resnet50_native->parameters(), /*lr=*/lr);

        model_resnet50_native->train();
        for (size_t epoch = 1; epoch <= epochs; ++epoch)
        {
            size_t batch_index = 0;
            double running_loss = 0.0;
            int running_corrects = 0;
            int num_samples = 0;
            cudaEventRecord(start);

            for (auto &batch : *train_dl_cifar10)
            {
                num_samples += batch.data.size(0);
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                optimizer_resnet50_native.zero_grad();

                torch::Tensor outputs = model_resnet50_native->forward(batch_data);
                torch::Tensor loss = torch::nll_loss(torch::log_softmax(outputs, /*dim=*/1), batch_target);

                loss.backward();
                optimizer_resnet50_native.step();
                running_loss += loss.item<double>();
                torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
                running_corrects += torch::sum(predictions == batch_target).item<int>();

                if (++batch_index % log_interval == 0)
                {
                    std::cout << "[" << epoch << "]\t[" << batch_index << "]\tLoss: " << loss.item<float>() << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << "NativeResNet-50,training,"
                         << epoch << "," << running_loss / (batch_index + 1) << ","
                         << (float)running_corrects / (float)num_samples << ","
                         << milliseconds << std::endl;
        }

        model_resnet50_native->eval();
        int batch_index = 0;
        double running_loss = 0.0;
        int corrects = 0;
        int num_samples = 0;

        {
            torch::NoGradGuard no_grad;

            cudaEventRecord(start);
            for (auto &batch : *test_dl_cifar10)
            {
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                torch::Tensor outputs = model_resnet50_native->forward(batch_data);
                torch::Tensor loss = torch::nll_loss(torch::log_softmax(outputs, /*dim=*/1), batch_target);
                torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
                int corrects = torch::sum(predictions == batch_target).item<int>();

                batch_index++;
                corrects += torch::sum(predictions == batch_target).item<int>();
                num_samples += batch.data.size(0);
                running_loss += loss.item<double>();
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
        results_file << "NativeResNet-50,inference,"
                     << 1 << "," << running_loss / (double)batch_index << ","
                     << (float)corrects / (float)num_samples << ","
                     << milliseconds << std::endl;

        {
            torch::NoGradGuard no_grad;

            int warmup = -latency_warmup_steps;
            bool warmup_finished = false;

            for (auto &batch : *test_dl_cifar10)
            {
                warmup_finished = warmup_finished || warmup > latency_warmup_steps;
                warmup += batch_size;

                torch::Tensor batch_data = batch.data.to(device);

                for (int i = 0; i < (i <= warmup_finished ? batch_size : epochs); i++)
                {
                    torch::Tensor sample = batch_data[i];
                    sample = sample.unsqueeze(0);

                    cudaEventRecord(start);
                    torch::Tensor outputs = model_resnet50_native->forward(batch_data);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    if (warmup_finished)
                        results_file << "NativeResNet-50,latency," << i + 1 << ",-1,-1,"
                                     << milliseconds << std::endl;
                }

                if (warmup_finished)
                    break;
            }
        }
    }

    if (RUN_TORCHSCRIPT) //======================================serailized PyTorch models
    {
        for (std::string model_name : models)
        {
            torch::jit::script::Module model_to_be_moved = torch::jit::load("./serialized_models/" + model_name + "_for_cifar10.pt");
            std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_to_be_moved);
            if (model == nullptr)
            {
                std::cout << "Error materializing a model from ./serialized_models/" + model_name + "_for_cifar10.pt" << std::endl;
                break;
            }

            std::cout << "Benchmarks for " << model_name << " begin." << std::endl;

            model->to(device);
            // workaround, since torch::jit::parameter_list is not supported by torch::optim::SGD
            std::vector<at::Tensor> model_parameters;
            for (const auto &params : model->parameters())
            {
                model_parameters.push_back(params);
            }
            torch::optim::SGD optimizer(model_parameters, /*lr=*/lr);

            model->train();
            for (size_t epoch = 1; epoch <= epochs; ++epoch)
            {
                size_t batch_index = 0;
                double running_loss = 0.0;
                int running_corrects = 0;
                int num_samples = 0;
                cudaEventRecord(start);

                for (auto &batch : *train_dl_cifar10)
                {
                    num_samples += batch.data.size(0);
                    torch::Tensor batch_data = batch.data.to(device);
                    torch::Tensor batch_target = batch.target.to(device);

                    optimizer.zero_grad();

                    // workaround, since torch::jit::script::Module::forward has different signature
                    std::vector<torch::jit::IValue> batch_data_ivalues;
                    batch_data_ivalues.push_back(batch_data);
                    torch::Tensor outputs = model->forward(batch_data_ivalues).toTensor();
                    torch::Tensor loss = torch::nll_loss(torch::log_softmax(outputs, /*dim=*/1), batch_target);

                    loss.backward();
                    optimizer.step();
                    running_loss += loss.item<double>();
                    torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
                    running_corrects += torch::sum(predictions == batch_target).item<int>();

                    if (++batch_index % log_interval == 0)
                    {
                        std::cout << "[" << epoch << "]\t[" << batch_index << "]\tLoss: " << loss.item<float>() << std::endl;
                    }
                }

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliseconds, start, stop);

                std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
                results_file << model_name << ",training,"
                             << epoch << "," << running_loss / (batch_index + 1) << ","
                             << (float)running_corrects / (float)num_samples << ","
                             << milliseconds << std::endl;
            }

            model->eval();
            int batch_index = 0;
            double running_loss = 0.0;
            int corrects = 0;
            int num_samples = 0;

            {
                torch::NoGradGuard no_grad;

                cudaEventRecord(start);
                for (auto &batch : *test_dl_cifar10)
                {
                    torch::Tensor batch_data = batch.data.to(device);
                    torch::Tensor batch_target = batch.target.to(device);

                    std::vector<torch::jit::IValue> batch_data_ivalues;
                    batch_data_ivalues.push_back(batch_data);
                    torch::Tensor outputs = model->forward(batch_data_ivalues).toTensor();
                    torch::Tensor loss = torch::nll_loss(torch::log_softmax(outputs, /*dim=*/1), batch_target);
                    torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
                    int corrects = torch::sum(predictions == batch_target).item<int>();

                    batch_index++;
                    corrects += torch::sum(predictions == batch_target).item<int>();
                    num_samples += batch.data.size(0);
                    running_loss += loss.item<double>();
                }

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliseconds, start, stop);
            }

            std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
            results_file << model_name << ",inference,"
                         << 1 << "," << running_loss / (double)batch_index << ","
                         << (float)corrects / (float)num_samples << ","
                         << milliseconds << std::endl;

            {
                torch::NoGradGuard no_grad;

                int warmup = -latency_warmup_steps;
                bool warmup_finished = false;

                for (auto &batch : *test_dl_cifar10)
                {
                    warmup_finished = warmup_finished || warmup > latency_warmup_steps;
                    warmup += batch_size;

                    torch::Tensor batch_data = batch.data.to(device);

                    for (int i = 0; i < (i <= warmup_finished ? batch_size : epochs); i++)
                    {
                        torch::Tensor sample = batch_data[i];
                        sample = sample.unsqueeze(0);
                        std::vector<torch::jit::IValue> sample_ivalue;
                        sample_ivalue.push_back(sample);

                        cudaEventRecord(start);
                        torch::jit::IValue outputs = model->forward(sample_ivalue);
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);
                        cudaEventElapsedTime(&milliseconds, start, stop);

                        if (warmup_finished)
                        {
                            std::cout << "latency: " << milliseconds << std::endl;
                            results_file << model_name << ",latency," << i + 1 << ",-1,-1,"
                                         << milliseconds << std::endl;
                        }
                    }

                    if (warmup_finished)
                        break;
                }
            }
        }
    }

    if (RUN_DCGAN) //================================================================DCGAN
    {
        std::chrono::steady_clock::time_point start_read = std::chrono::steady_clock::now();
        auto celeba = CELEBA{"../datasets/celeba_tiny", batch_size};
        std::chrono::steady_clock::time_point end_read = std::chrono::steady_clock::now();

        results_file << "CELEBA,read,1,-1,-1,"
                     << std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count()
                     << std::endl;

        std::cout << "Benchmarks for DCGAN begin." << std::endl;
        int latent_vec_size = 100;
        int gen_batch_size = 1024;
        auto generator = std::make_shared<Generator>();
        auto discriminator = std::make_shared<Discriminator>();

        generator->to(device);
        discriminator->to(device);

        torch::optim::Adam gen_optimizer(generator->parameters(),
                                         torch::optim::AdamOptions(lr).betas(std::make_tuple(0.5, 0.999)));
        torch::optim::Adam disc_optimizer(discriminator->parameters(),
                                          torch::optim::AdamOptions(lr).betas(std::make_tuple(0.5, 0.999)));

        generator->train();
        discriminator->train();

        for (size_t epoch = 1; epoch <= epochs; ++epoch)
        {
            std::cout << "Epoch " << epoch << " begins." << std::endl;
            size_t batch_index = 0;
            double running_loss_G = 0.0, running_loss_D = 0.0;
            double running_D_x = 0.0, running_D_G_z1 = 0.0, running_D_G_z2 = 0.0;
            double real_label = 1.0, fake_label = 0.0;
            cudaEventRecord(start);

            int max_batch = celeba.get_max_batch_id();
            while (batch_index < max_batch)
            {
                disc_optimizer.zero_grad();
                torch::Tensor real_cpu = celeba.get_batch_by_id(batch_index).to(device);
                int b_size = real_cpu.size(0);
                auto label = torch::full({b_size}, real_label,
                                         torch::TensorOptions().dtype(torch::kFloat).device(device));

                torch::Tensor output = discriminator->forward(real_cpu).view(-1);
                torch::Tensor errD_real = torch::binary_cross_entropy(output, label);
                errD_real.backward();
                auto D_x = output.mean().item();

                torch::Tensor noise = torch::randn({b_size, latent_vec_size, 1, 1},
                                                   torch::TensorOptions().device(device));
                torch::Tensor fake = generator->forward(noise);
                label.fill_(fake_label);
                output = discriminator->forward(fake.detach()).view(-1);
                torch::Tensor errD_fake = torch::binary_cross_entropy(output, label);
                errD_fake.backward();
                auto D_G_z1 = output.mean().item();
                auto errD = errD_real + errD_fake;
                disc_optimizer.step();

                gen_optimizer.zero_grad();
                label.fill_(real_label);
                output = discriminator->forward(fake).view(-1);
                torch::Tensor errG = torch::binary_cross_entropy(output, label);
                errG.backward();
                auto D_G_z2 = output.mean().item();
                gen_optimizer.step();

                // telemetry
                running_loss_G += errG.item<double>();
                running_loss_D += errD.item<double>();
                running_D_x += D_x.toDouble();
                running_D_G_z1 += D_G_z1.toDouble();
                running_D_G_z2 += D_G_z2.toDouble();

                if (++batch_index % log_interval == 0)
                {
                    std::cout << "[" << epoch << "]\t[" << batch_index << " / " << max_batch
                              << "]\tLoss_G: " << errG.item<double>()
                              << "\tLoss_D: " << errD.item<double>()
                              << "\tD(x): " << D_x.toDouble()
                              << "\tD(G(z)): " << D_G_z1.toDouble() << " / " << D_G_z2.toDouble()
                              << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            running_loss_G /= (batch_index + 1);
            running_loss_D /= (batch_index + 1);
            running_D_x /= (batch_index + 1);
            running_D_G_z1 /= (batch_index + 1);
            running_D_G_z2 /= (batch_index + 1);

            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << "DCGAN,training,"
                         << epoch << ","
                         << running_loss_G << "|" << running_loss_D << ","
                         << running_D_x << "|" << running_D_G_z1 << "|" << running_D_G_z2 << ","
                         << milliseconds << std::endl;
        }

        generator->eval();
        discriminator->eval();

        torch::Tensor latent_vecs_batch = torch::randn(
            {gen_batch_size, latent_vec_size, 1, 1}, torch::TensorOptions().device(device));

        {
            torch::NoGradGuard no_grad;

            cudaEventRecord(start);
            torch::Tensor res_images = generator->forward(latent_vecs_batch);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        std::cout << "Generation time: " << milliseconds << " ms" << std::endl;
        results_file << "DCGAN,generation,1,-1,-1," << milliseconds << std::endl;

        {
            torch::NoGradGuard no_grad;

            for (int i = 1; i <= latency_warmup_steps + epochs; i++)
            {
                torch::Tensor sample = torch::randn(
                    {1, latent_vec_size, 1, 1}, torch::TensorOptions().device(device));

                cudaEventRecord(start);
                torch::Tensor outputs = generator->forward(sample);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliseconds, start, stop);

                if (i > latency_warmup_steps)
                    results_file << "DCGAN,latency," << i + 1 << ",-1,-1,"
                                 << milliseconds << std::endl;
            }
        }
    }

    if (RUN_SODNET) //==============================================================SODNet
    {
        std::cout << "Benchmarks for SODNet begin." << std::endl;
        auto model_sodnet = std::make_shared<SODNet>(SODNet(3, 16));
        model_sodnet->to(device);
        torch::optim::SGD optimizer_sodnet(model_sodnet->parameters(), /*lr=*/lr);

        model_sodnet->train();
        for (size_t epoch = 1; epoch <= epochs; ++epoch)
        {
            size_t batch_index = 0;
            double running_loss = 0.0;
            int num_samples = 0;
            cudaEventRecord(start);

            for (auto &batch : *train_dl_adam)
            {
                num_samples += batch.data.size(0);
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                optimizer_sodnet.zero_grad();

                torch::Tensor outputs = model_sodnet->forward(batch_data);
                torch::Tensor loss = torch::smooth_l1_loss(outputs, batch_target, at::Reduction::Sum);

                loss.backward();
                optimizer_sodnet.step();
                running_loss += loss.item<double>();

                if (++batch_index % log_interval == 0)
                {
                    std::cout << "[" << epoch << "]\t[" << batch_index << "]\tLoss: " << loss.item<float>() << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << "SODNet,training,"
                         << epoch << "," << running_loss / (batch_index + 1) << ","
                         << -1 /* IoU neasured only for PyTorch impl :) */ << ","
                         << milliseconds << std::endl;
        }

        model_sodnet->eval();
        int batch_index = 0;
        double running_loss = 0.0;

        {
            torch::NoGradGuard no_grad;

            cudaEventRecord(start);
            for (auto &batch : *test_dl_adam)
            {
                torch::Tensor batch_data = batch.data.to(device);
                torch::Tensor batch_target = batch.target.to(device);

                torch::Tensor outputs = model_sodnet->forward(batch_data);
                torch::Tensor loss = torch::smooth_l1_loss(outputs, batch_target);

                batch_index++;
                running_loss += loss.item<double>();
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        std::cout << "Eval time: " << milliseconds << " ms" << std::endl;
        results_file << "SODNet,detection,"
                     << 1 << "," << running_loss / (double)batch_index << ","
                     << -1 /* idc abt IoU dude !!! */ << ","
                     << milliseconds << std::endl;

        {
            torch::NoGradGuard no_grad;

            int warmup = -latency_warmup_steps;
            bool warmup_finished = false;

            for (auto &batch : *test_dl_adam)
            {
                warmup_finished = warmup_finished || warmup > latency_warmup_steps;
                warmup += batch_size;

                torch::Tensor batch_data = batch.data.to(device);

                for (int i = 0; i < (i <= warmup_finished ? batch_size : epochs); i++)
                {
                    torch::Tensor sample = batch_data[i];
                    sample = sample.unsqueeze(0);

                    cudaEventRecord(start);
                    torch::Tensor outputs = model_sodnet->forward(batch_data);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    if (warmup_finished)
                        results_file << "SODNet,latency," << i + 1 << ",-1,-1,"
                                     << milliseconds << std::endl;
                }

                if (warmup_finished)
                    break;
            }
        }
    }

    results_file.close();
    return 0;
}