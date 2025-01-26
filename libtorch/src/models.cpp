#include "models.h"
#include <torch/torch.h>
#include <string>

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

SimpleConvNet::SimpleConvNet(int num_classes)
{
	torch::nn::Sequential conv1_unregistered{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
		// torch::nn::BatchNorm2d(16),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

	torch::nn::Sequential conv2_unregistered{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
		// torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

	torch::nn::Linear dense_unregistered = torch::nn::Linear(32 * 7 * 7, 512);
	torch::nn::Linear classifier_unregistered = torch::nn::Linear(512, num_classes);

	conv1 = register_module("conv1", conv1_unregistered);
	conv2 = register_module("conv2", conv2_unregistered);
	dense = register_module("dense", dense_unregistered);
	classifier = register_module("classifier", classifier_unregistered);
}

torch::Tensor SimpleConvNet::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = conv2->forward(x);
	x = x.view({-1, 32 * 7 * 7});
	// x = x.flatten(1);
	x = torch::relu(dense->forward(x));
	return torch::log_softmax(classifier->forward(x), /*dim=*/1);
}

ExtendedConvNet::ExtendedConvNet(int num_classes)
{
	torch::nn::Sequential conv1_unregistered{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

	torch::nn::Sequential conv2_unregistered{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

	torch::nn::Sequential extended_convs_unregistered{
		// ec1
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(5)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		// ec2
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(5)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		// ec3
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(5)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		// ec4
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(5)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		// ec5
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(5)),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
	};

	torch::nn::Linear dense_unregistered = torch::nn::Linear(32 * 7 * 7, 512);
	torch::nn::Sequential extended_dense_unregistered{
		// ed1
		torch::nn::Linear(512, 256),
		torch::nn::ReLU(),
		// ed2
		torch::nn::Linear(256, 128),
		torch::nn::ReLU(),
		// ed3
		torch::nn::Linear(128, 64),
		torch::nn::ReLU(),
		// ed4
		torch::nn::Linear(64, 32),
		torch::nn::ReLU(),
		// ed5
		torch::nn::Linear(32, 16),
		torch::nn::ReLU()
	};
	torch::nn::Linear classifier_unregistered = torch::nn::Linear(16, num_classes);

	conv1 = register_module("conv1", conv1_unregistered);
	conv2 = register_module("conv2", conv2_unregistered);
	extended_convs = register_module("extended_convs", extended_convs_unregistered);
	dense = register_module("dense", dense_unregistered);
	extended_dense = register_module("extended_dense", extended_dense_unregistered);
	classifier = register_module("classifier", classifier_unregistered);
}

torch::Tensor ExtendedConvNet::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = conv2->forward(x);
	x = extended_convs->forward(x);
	x = x.flatten(1);
	x = torch::relu(dense->forward(x));
	x = extended_dense->forward(x);
	return torch::log_softmax(classifier->forward(x), /*dim=*/1);
}

Bottleneck::Bottleneck(
	int in_channels,
	int intermediate_channels,
	int expansion,
	bool is_bottleneck,
	int stride)
{
	this->in_channels = in_channels;
	this->intermediate_channels = intermediate_channels;
	this->expansion = expansion;
	this->is_bottleneck = is_bottleneck;

	if (in_channels == intermediate_channels * expansion)
	{
		this->identity = true;
	}
	else
	{
		this->identity = false;
		torch::nn::Sequential projection_unreg{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(in_channels, intermediate_channels * expansion, 1)
					.stride(stride)
					.padding(0)
					.bias(false)),
			torch::nn::BatchNorm2d(intermediate_channels * expansion)};
		this->projection = register_module("projection", projection_unreg);
	}

	if (is_bottleneck)
	{
		// bottleneck
		// 1x1
		torch::nn::Sequential conv1_1x1_unreg{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(in_channels, intermediate_channels, 1)
					.stride(1)
					.padding(0)
					.bias(false)),
			torch::nn::BatchNorm2d(intermediate_channels),
			torch::nn::ReLU()};

		// 3x3
		torch::nn::Sequential conv2_3x3_unreg{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(intermediate_channels, intermediate_channels, 3)
					.stride(stride)
					.padding(1)
					.bias(false)),
			torch::nn::BatchNorm2d(intermediate_channels),
			torch::nn::ReLU()};

		// 1x1
		torch::nn::Sequential conv2_1x1_unreg{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(intermediate_channels, intermediate_channels * expansion, 1)
					.stride(1)
					.padding(0)
					.bias(false)),
			torch::nn::BatchNorm2d(intermediate_channels * expansion)};

		this->conv1_1x1 = register_module("conv1_1x1", conv1_1x1_unreg);
		this->conv2_3x3 = register_module("conv2_3x3", conv2_3x3_unreg);
		this->conv2_1x1 = register_module("conv2_1x1", conv2_1x1_unreg);
	}
	else
	{
		// basic block
		// 3x3
		torch::nn::Sequential conv1_3x3_unreg{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(in_channels, intermediate_channels, 3)
					.stride(stride)
					.padding(1)
					.bias(false)),
			torch::nn::BatchNorm2d(intermediate_channels),
			torch::nn::ReLU()};

		// 3x3
		torch::nn::Sequential conv2_3x3_unreg{
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(intermediate_channels, intermediate_channels, 3)
					.stride(1)
					.padding(1)
					.bias(false)),
			torch::nn::BatchNorm2d(intermediate_channels)};

		this->conv1_3x3 = register_module("conv1_3x3", conv1_3x3_unreg);
		this->conv2_3x3 = register_module("conv2_3x3", conv2_3x3_unreg);
	}
}

torch::Tensor Bottleneck::forward(torch::Tensor x)
{
	torch::Tensor in_x = x.detach().clone();

	if (this->is_bottleneck)
	{
		x = conv1_1x1->forward(x);
		x = conv2_3x3->forward(x);
		x = conv2_1x1->forward(x);
	}
	else
	{
		x = conv1_3x3->forward(x);
		x = conv2_3x3->forward(x);
	}

	if (this->identity)
	{
		x += in_x;
	}
	else
	{
		x += projection->forward(in_x);
	}

	x = torch::nn::ReLU()->forward(x);
	return x;
}

ResNet50::ResNet50(int num_classes)
{
	// ResNet-50 parameters as specified in the white paper
	int channels_list[] = {64, 128, 256, 512};
	int repeatition_list[] = {3, 4, 6, 3};
	int expansion = 4;
	int is_bottleneck = true;

	torch::nn::Sequential process_input_unreg{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)),
		torch::nn::BatchNorm2d(64),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1))};
	this->process_input = register_module("process_input", process_input_unreg);

	torch::nn::Sequential block1_unreg{
		Bottleneck(64, channels_list[0], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[0] * expansion, channels_list[0], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[0] * expansion, channels_list[0], expansion, is_bottleneck, 1)};
	this->block1 = register_module("block1", block1_unreg);

	torch::nn::Sequential block2_unreg{
		Bottleneck(channels_list[0] * expansion, channels_list[1], expansion, is_bottleneck, 2),
		Bottleneck(channels_list[1] * expansion, channels_list[1], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[1] * expansion, channels_list[1], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[1] * expansion, channels_list[1], expansion, is_bottleneck, 1)};
	this->block2 = register_module("block2", block2_unreg);

	torch::nn::Sequential block3_unreg{
		Bottleneck(channels_list[1] * expansion, channels_list[2], expansion, is_bottleneck, 2),
		Bottleneck(channels_list[2] * expansion, channels_list[2], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[2] * expansion, channels_list[2], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[2] * expansion, channels_list[2], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[2] * expansion, channels_list[2], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[2] * expansion, channels_list[2], expansion, is_bottleneck, 1)};
	this->block3 = register_module("block3", block3_unreg);

	torch::nn::Sequential block4_unreg{
		Bottleneck(channels_list[2] * expansion, channels_list[3], expansion, is_bottleneck, 2),
		Bottleneck(channels_list[3] * expansion, channels_list[3], expansion, is_bottleneck, 1),
		Bottleneck(channels_list[3] * expansion, channels_list[3], expansion, is_bottleneck, 1)};
	this->block4 = register_module("block4", block4_unreg);

	this->average_pool = register_module("average_pool",
										 torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)));

	this->fc1 = register_module("fc1", torch::nn::Linear(channels_list[3] * expansion, num_classes));
}

torch::Tensor ResNet50::forward(torch::Tensor x)
{
	x = process_input->forward(x);
	x = block1->forward(x);
	x = block2->forward(x);
	x = block3->forward(x);
	x = block4->forward(x);
	x = average_pool->forward(x);
	x = torch::flatten(x, 1);
	x = fc1->forward(x);
	return x;
}

Generator::Generator(int n_channels, int latent_vec_size, int feat_map_size)
{
	torch::nn::Sequential main_unreg{
		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(latent_vec_size, feat_map_size * 8, 4).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size * 8)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),

		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(feat_map_size * 8, feat_map_size * 4, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size * 4)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),

		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(feat_map_size * 4, feat_map_size * 2, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size * 2)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),

		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(feat_map_size * 2, feat_map_size, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),

		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(feat_map_size, n_channels, 4).stride(2).padding(1).bias(false)),
		torch::nn::Tanh()
	};
	this->main = register_module("main", main_unreg);
}

torch::Tensor Generator::forward(torch::Tensor x)
{
	return this->main->forward(x);
}

Discriminator::Discriminator(int n_channels, int feat_map_size)
{
	torch::nn::Sequential main_unreg{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(n_channels, feat_map_size, 4).stride(2).padding(1).bias(false)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(feat_map_size, feat_map_size * 2, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size * 2)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(feat_map_size * 2, feat_map_size * 4, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size * 4)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(feat_map_size * 4, feat_map_size * 8, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(feat_map_size * 8)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(feat_map_size * 8, 1, 4).stride(1).padding(0).bias(false)),
		torch::nn::Sigmoid()
	};
	this->main = register_module("main", main_unreg);
}

torch::Tensor Discriminator::forward(torch::Tensor x)
{
	return this->main->forward(x);
}

ResBlock::ResBlock(int in_channels, int out_channels)
{
	torch::nn::Sequential base1_unreg{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(1).padding(1)),
		torch::nn::BatchNorm2d(in_channels),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
	};

	torch::nn::Sequential base2_unreg{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)),
		torch::nn::BatchNorm2d(out_channels),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
	};

	this->base1 = register_module("base1", base1_unreg);
	this->base2 = register_module("base2", base2_unreg);
	this->mpool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
}

torch::Tensor ResBlock::forward(torch::Tensor x)
{
	x = this->base1->forward(x) + x;
	x = this->base2->forward(x);
	x = this->mpool->forward(x);
	return x;
}

SODNet::SODNet(int in_channels, int first_output_channels)
{
	torch::nn::Sequential main_unreg{
		ResBlock(in_channels, first_output_channels),
		ResBlock(first_output_channels, 2 * first_output_channels),
		ResBlock(2 * first_output_channels, 4 * first_output_channels),
		ResBlock(4 * first_output_channels, 8 * first_output_channels),		

		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(
				8 * first_output_channels, 16 * first_output_channels, 3)),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),

		torch::nn::Flatten(),
		torch::nn::Linear(7 * 7 * 16 * first_output_channels, 2)
	};

	this->main = register_module("main", main_unreg);
}

torch::Tensor SODNet::forward(torch::Tensor x)
{
	return this->main->forward(x);
}