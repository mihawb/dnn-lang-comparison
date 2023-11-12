import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50, densenet121, mobilenet_v2, convnext_small
import torch.optim as optim
from torch_funcs import fit, test, get_cifar10_loaders, get_mnist_loaders, get_celeba_loader, FullyConnectedNet, SimpleConvNet, fit_dcgan, Generator, Discriminator, dcgan_weights_init, generate
import pandas as pd
import numpy as np


batch_size = 96
test_batch_size = 128
epochs = 15
lr = 1e-2
momentum = 0.9
num_classes = 10
log_interval = 200
nc = 3
nz = 100
ngf = 64
ndf = 64
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'CUDA enabled: {use_cuda}')


def env_builder(name: str): 
	if name == 'FullyConnectedNet':
		model = FullyConnectedNet()
	elif name == 'SimpleConvNet':
		model = SimpleConvNet()
	elif name == 'ResNet-50':
		model = resnet50()
		model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
	elif name == 'DenseNet-121':
		model = densenet121()
		model.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
	elif name == 'MobileNet-v2':
		model = mobilenet_v2()
		model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
	elif name == 'ConvNeXt-Small':
		model = convnext_small()
		model.classifier[2] = nn.Linear(in_features=768, out_features=num_classes, bias=True)
	else:
		raise ValueError('Invalid model name')

	if name == 'FullyConnectedNet':
		train_dl, _, test_dl = get_mnist_loaders(batch_size, test_batch_size)
		loss_func = F.nll_loss
	elif name == 'SimpleConvNet':
		train_dl, _, test_dl = get_mnist_loaders(batch_size, test_batch_size, flatten=False)
		loss_func = F.nll_loss
	else:
		train_dl, test_dl = get_cifar10_loaders(batch_size, test_batch_size)
		loss_func = F.cross_entropy

	return model, train_dl, test_dl, loss_func


telemetry = {
	'model_name': [],
	'type':[],
	'epoch': [],
	'loss': [],
	'performance': [],
	'elapsed_time': []
}


if __name__ == '__main__':
	for model_name in ['FullyConnectedNet', 'SimpleConvNet', 'ResNet-50', 'DenseNet-121', 'MobileNet-v2', 'ConvNeXt-Small']:
		print(f'Benchmarks for {model_name} begin')

		model, train_dl, test_dl, loss_func = env_builder(model_name)
		model = model.to(device)
		opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

		# training
		for epoch in range(1, epochs + 1):
			start.record()
			train_history = fit(model, device, train_dl, loss_func, epoch, optimizer=opt, log_interval=log_interval, silent=False)
			end.record()
			torch.cuda.synchronize()

			_, accuracy = test(model, device, test_dl, loss_func, silent=True)

			telemetry['model_name'].append(model_name)
			telemetry['type'].append('training')
			telemetry['epoch'].append(epoch)
			telemetry['loss'].append(train_history[-1])
			telemetry['performance'].append(accuracy)
			telemetry['elapsed_time'].append(start.elapsed_time(end))

		# inference
		start.record()
		loss, accuracy = test(model, device, test_dl, loss_func, silent=True)
		end.record()
		torch.cuda.synchronize()

		telemetry['model_name'].append(model_name)
		telemetry['type'].append('inference')
		telemetry['epoch'].append(1)
		telemetry['loss'].append(loss)
		telemetry['performance'].append(accuracy)
		telemetry['elapsed_time'].append(start.elapsed_time(end))
		pd.DataFrame(telemetry).to_csv(f'../results/pytorch_results.csv', index=False)

		del model

	netG = Generator(nc, nz, ngf).to(device)
	netD = Discriminator(nc, ndf).to(device)
	netG.apply(dcgan_weights_init)
	netD.apply(dcgan_weights_init)

	celeba_dl = get_celeba_loader(batch_size=batch_size)

	loss_func = nn.BCELoss()
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

	print('Benchmarks for DCGAN begin')
	for epoch in range(1, epochs + 1):
		start.record()
		gan_hist = fit_dcgan(netG, netD, device, celeba_dl, loss_func, epoch, optimizerG, optimizerD, nz, log_interval=log_interval)
		end.record()
		torch.cuda.synchronize()

		for stat in gan_hist:
			gan_hist[stat] = np.sum(gan_hist[stat]) / len(gan_hist[stat])

		telemetry['model_name'].append('DCGAN')
		telemetry['type'].append('training')
		telemetry['epoch'].append(epoch)
		telemetry['loss'].append(f'{gan_hist["loss_G"]}|{gan_hist["loss_D"]}')
		telemetry['performance'].append(f'{gan_hist["D_x"]}|{gan_hist["D_G_z1"]}|{gan_hist["D_G_z2"]}')
		telemetry['elapsed_time'].append(start.elapsed_time(end))
		pd.DataFrame(telemetry).to_csv(f'../results/pytorch_results.csv', index=False)

	# generation
	start.record()
	_ = generate(netG, device, 1, test_batch_size=test_batch_size)
	end.record()
	torch.cuda.synchronize()

	telemetry['model_name'].append('DCGAN')
	telemetry['type'].append('generation')
	telemetry['epoch'].append(1)
	telemetry['loss'].append(-1)
	telemetry['performance'].append(-1)
	telemetry['elapsed_time'].append(start.elapsed_time(end))
	pd.DataFrame(telemetry).to_csv(f'../results/pytorch_results.csv', index=False)
