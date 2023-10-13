#include "models.h"
#include <torch/torch.h>

FullyConnectedNet::FullyConnectedNet()
{
	// Construct and register two Linear submodules.
	fc1 = register_module("fc1", torch::nn::Linear(784, 64));
	fc2 = register_module("fc2", torch::nn::Linear(64, 32));
	fc3 = register_module("fc3", torch::nn::Linear(32, 10));
}

torch::Tensor FullyConnectedNet::forward(torch::Tensor x)
{
	x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
	x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
	x = torch::relu(fc2->forward(x));
	x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
	return x;
}