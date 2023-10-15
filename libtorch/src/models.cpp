#include "models.h"
#include <torch/torch.h>

FullyConnectedNet::FullyConnectedNet(int num_classes)
{
	// Construct and register two Linear submodules.
	fc1 = register_module("fc1", torch::nn::Linear(784, 64));
	fc2 = register_module("fc2", torch::nn::Linear(64, 32));
	fc3 = register_module("fc3", torch::nn::Linear(32, num_classes));
}

torch::Tensor FullyConnectedNet::forward(torch::Tensor x)
{
	x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
	x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
	x = torch::relu(fc2->forward(x));
	x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
	return x;
}

SimpleConvNet::SimpleConvNet(int num_classes) {
	torch::nn::Sequential conv1_unregistered {
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
		// torch::nn::BatchNorm2d(16),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

	torch::nn::Sequential conv2_unregistered {
		torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
		// torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

	torch::nn::Linear dense_unregistered = torch::nn::Linear(32 * 7 * 7, 500);
	torch::nn::Linear classifier_unregistered = torch::nn::Linear(500, num_classes);

	conv1 = register_module("conv1", conv1_unregistered);
	conv2 = register_module("conv2", conv2_unregistered);
	dense = register_module("dense", dense_unregistered);
	classifier = register_module("classifier", classifier_unregistered);
}

torch::Tensor SimpleConvNet::forward(torch::Tensor x) {
	x = conv1->forward(x);
	x = conv2->forward(x);
	x = x.view({-1, 32 * 7 * 7});
	x = torch::relu(dense->forward(x));
	return torch::log_softmax(classifier->forward(x), /*dim=*/1);
}