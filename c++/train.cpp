#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"
#include "src/builders.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <fstream>

using namespace cudl;

int main(int argc, char* argv[])
{
    int which_model;
    std::string model_name = argc > 1 ? argv[1] : "__invalid__";

    if (model_name.compare("fcnet") == 0)
        which_model = 0;
    else if (model_name.compare("scvnet") == 0)
        which_model = 1;
    else {
        std::cout << "Invalid model name.\nChoose one of the following:\nfcnet, scvnet" << std::endl;
        return 1; 
    }

    std::cout << "Using model: " << model_name << std::endl;

    /* configure the network */
    int batch_size_train = 32;
    int num_steps_in_ep_train = 60000 / batch_size_train;
    int monitoring_step = 200;
    int epochs = 15;

    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 64;
    int num_steps_test = 1000 / batch_size_test;

    std::ofstream results_file;
    results_file.open("../results/cpp_"+model_name+".csv", std::ios::out);
    results_file << "mnames,type,eps,trloss,acc,times" << std::endl;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("../datasets/mnist-digits");
    train_data_loader.train(batch_size_train, true);

    // step 2. model initialization
    Network model;
    ModelFactory(model, which_model);
    model.cuda();

    if (load_pretrain)
        model.load_pretrain();
    model.train();

    // step 3. train
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        int tp_count = 0;
        float loss;
        float accuracy;

        int step = train_data_loader.train_reset();
        train_data_loader.get_batch();

        cudaEventRecord(start);

        while (step < num_steps_in_ep_train)
        {
            // nvtx profiling start
            std::string nvtx_message = std::string("step" + std::to_string(step));
            nvtxRangePushA(nvtx_message.c_str());

            // update shared buffer contents
            train_data->to(cuda);
            train_target->to(cuda);
            
            // forward
            model.forward(train_data);
            tp_count += model.get_accuracy(train_target);

            // back-propagation
            model.backward(train_target);

            // update parameter
            // we will use learning rate decay to the learning rate
            learning_rate *= 1.f / (1.f + lr_decay * step);
            model.update(learning_rate);

            // fetch next data
            step = train_data_loader.next();

            // nvtx profiling end
            nvtxRangePop();

            // calculation softmax loss
            if (step % monitoring_step == 0)
            {
                loss = model.loss(train_target);
                accuracy =  100.f * tp_count / monitoring_step / batch_size_train;
                
                std::cout << "epoch: " << std::right << std::setw(2) << epoch << \
                            ", step: " << std::right << std::setw(4) << step << \
                            ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                            ", accuracy: " << accuracy << "%" << std::endl;

                tp_count = 0;
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Epoch time: " << milliseconds << " ms" << std::endl;

        results_file << model_name << ",training," << epoch << "," << loss << "," << accuracy << "," << milliseconds << std::endl;
    }

    // trained parameter save
    if (file_save)
        model.write_file();

    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("../datasets/mnist-digits");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    model.test();
    
    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    int tp_count = 0;
    int step = 0;
    cudaEventRecord(start);
    while (step < num_steps_test)
    {

        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
		test_data->to(cuda);
		test_target->to(cuda);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
    }    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Inference time: " << milliseconds << " ms" << std::endl;

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;
    results_file << model_name << ",inference,1," << loss << "," << accuracy << "," << milliseconds << std::endl;

    // Good bye
    std::cout << "Done." << std::endl;
    results_file.close();

    return 0;
}
