#include "builders.h"
#include "network.h"
#include "layer.h"

// using namespace cudl;

void cudl::FullyConnectedNetBuilder(Network &model)
{
	model.add_layer(new Dense("dense1", 800));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Dense("dense2", 10));
	model.add_layer(new Softmax("softmax"));
}

void cudl::SimpleConvNetBuilder(Network &model)
{
	model.add_layer(new Conv2D("conv1", 16, 5, 1, 2));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));

	model.add_layer(new Conv2D("conv2", 32, 5, 1, 2));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));

	model.add_layer(new Dense("dense1", 512));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

	model.add_layer(new Dense("dense2", 10));
	model.add_layer(new Softmax("softmax"));
}

void cudl::ExtendedConvNetBuilder(Network &model)
{
	// conv1
	model.add_layer(new Conv2D("conv1", 16, 5, 1, 2));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));

	// conv2
	model.add_layer(new Conv2D("conv2", 32, 5, 1, 2));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));

	// ec1
	model.add_layer(new Conv2D("ec1", 32, 3, 1, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
	// ec2
	model.add_layer(new Conv2D("ec2", 32, 3, 1, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
	// ec3
	model.add_layer(new Conv2D("ec3", 32, 3, 1, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
	// ec4
	model.add_layer(new Conv2D("ec4", 32, 3, 1, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
	// ec5
	model.add_layer(new Conv2D("ec5", 32, 3, 1, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));

	model.add_layer(new Dense("dense1", 512));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

	// ed 1
	model.add_layer(new Dense("ed2", 256));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	// ed 2
	model.add_layer(new Dense("ed3", 128));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	// ed 3
	model.add_layer(new Dense("ed4", 64));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	// ed 4
	model.add_layer(new Dense("ed4", 32));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	// ed 5
	model.add_layer(new Dense("ed1", 16));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

	model.add_layer(new Dense("classifier", 10));
	model.add_layer(new Softmax("softmax"));
}

// m = 0 | default	-> FCNet
// m = 1 			-> SimpleConvNet
// m = 2 			-> ExtendedConvNet
void cudl::ModelFactory(Network &model, int m)
{
	if (m == 1)
		SimpleConvNetBuilder(model);
	else if (m == 2)
		ExtendedConvNetBuilder(model);
	else
		FullyConnectedNetBuilder(model);
}
