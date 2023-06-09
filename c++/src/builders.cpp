#include "builders.h"
#include "network.h"
#include "layer.h"

// using namespace cudl;

void cudl::FullyConnectedNetBuilder(Network &model) {
    model.add_layer(new Dense("dense1", 800));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense("dense2", 10));
    model.add_layer(new Softmax("softmax"));
}

void cudl::SimpleConvNetBuilder(Network &model) {
		model.add_layer(new Conv2D("conv1", 20, 5));
		model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
		model.add_layer(new Conv2D("conv2", 50, 5));
		model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
		model.add_layer(new Dense("dense1", 500));
		model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
		model.add_layer(new Dense("dense2", 10));
		model.add_layer(new Softmax("softmax"));
}

// m = 0 | default	-> FCNet
// m = 1 						-> SimpleConvNet
// m = 2 						-> MobileNet (maybe later on)
void cudl::ModelFactory(Network &model, int m) {
	if (m == 1) 
		SimpleConvNetBuilder(model);
	else
		FullyConnectedNetBuilder(model);
}
