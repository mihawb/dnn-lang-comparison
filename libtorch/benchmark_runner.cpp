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

int main_for_tests() {
    // testy xdd
    torch::Tensor x = torch::randn({1,64,112,112});
    auto model_to_be_moved = Bottleneck(64,64,4,true,2);
    auto model = std::make_shared<Bottleneck>(model_to_be_moved);
    torch::Tensor out = model->forward(x);
    std::cout << torch::_shape_as_tensor(out) << std::endl << std::endl;

    torch::Tensor y = torch::randn({1,3,32,32});
    auto resnet_to_be_moved = ResNet50(10);
    auto resnet = std::make_shared<ResNet50>(resnet_to_be_moved);
    torch::Tensor pred = resnet->forward(y);
    std::cout << pred << std::endl;

    return 0;
}

// int main() {
//     auto resnet_to_be_moved = ResNet50(10);
//     auto resnet = std::make_shared<ResNet50>(resnet_to_be_moved);

//     resnet->parameters()

//     return 0;
// }

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
    results_file.open("../results/libtorch.csv", std::ios::out);
    results_file << "model_name,type,epoch,loss,performance,elapsed_time" << std::endl;

    int batch_size = 96;
    int test_batch_size = 128;
    int epochs = 15;
    float lr = 0.01;
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

    //===================================================================FullyConnectedNet
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

    //=======================================================================SimpleConvNet
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

    //===========================================================================ResNet-50
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
                std::cout << "[" << epoch << "]\t[" << batch_index <<
                    "]\tLoss: " << loss.item<float>() << std::endl;
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);


        std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
        results_file << "NativeResNet50,training," 
            << epoch << "," << running_loss / (batch_index+1) << ","
            << (float)running_corrects / (float)num_samples << ","
            << milliseconds << std::endl;
    }

    model_resnet50_native->eval();
    for (auto &batch : *test_dl_cifar10) {
        cudaEventRecord(start);

        torch::Tensor batch_data = batch.data.to(device);
        torch::Tensor batch_target = batch.target.to(device);

        torch::Tensor outputs = model_resnet50_native->forward(batch_data);
        torch::Tensor loss = torch::nll_loss(torch::log_softmax(outputs, /*dim=*/1), batch_target);
        torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
        int corrects = torch::sum(predictions == batch_target).item<int>();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        results_file << "NativeResNet50,inference," 
            << 1 << "," << loss.item<float>() << ","
            << (float)corrects / (float)batch.data.size(0) << ","
            << milliseconds << std::endl;
        break;
    }

    //===========================================================serailized PyTorch models
    std::string models[] = {"ResNet-50", "DenseNet-121", "MobileNet-v2", "ConvNeXt-Small"};

    for (std::string model_name : models) {
        torch::jit::script::Module model_to_be_moved = torch::jit::load("./serialized_models/" + model_name + "_for_cifar10.pt");
        std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_to_be_moved);
        if (model == nullptr) {
            std::cout << "Error materializing a model from ./serialized_models/" + model_name + "_for_cifar10.pt" << std::endl;
            break;
        }

        std::cout << "Benchmarks for " << model_name << " begin." << std::endl;

        model->to(device);
        // workaround, since torch::jit::parameter_list is not supported by torch::optim::SGD
        std::vector<at::Tensor> model_parameters;
        for (const auto& params : model->parameters()) {
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
                    std::cout << "[" << epoch << "]\t[" << batch_index <<
                        "]\tLoss: " << loss.item<float>() << std::endl;
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);


            std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;
            results_file << model_name << ",training," 
                << epoch << "," << running_loss / (batch_index+1) << ","
                << (float)running_corrects / (float)num_samples << ","
                << milliseconds << std::endl;
        }

        model->eval();
        for (auto &batch : *test_dl_cifar10) {
            cudaEventRecord(start);

            torch::Tensor batch_data = batch.data.to(device);
            torch::Tensor batch_target = batch.target.to(device);

            std::vector<torch::jit::IValue> batch_data_ivalues;
            batch_data_ivalues.push_back(batch_data);
            torch::Tensor outputs = model->forward(batch_data_ivalues).toTensor();
            torch::Tensor loss = torch::nll_loss(torch::log_softmax(outputs, /*dim=*/1), batch_target);
            torch::Tensor predictions = std::get<1>(torch::max(outputs, 1));
            int corrects = torch::sum(predictions == batch_target).item<int>();
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            results_file << model_name << ",inference," 
                << 1 << "," << loss.item<float>() << ","
                << (float)corrects / (float)batch.data.size(0) << ","
                << milliseconds << std::endl;
            break;
        }
    }

    results_file.close();
    return 0;
}