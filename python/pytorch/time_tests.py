from sodnet_funcs import fit_sodnet, test_sodnet, get_adam_loaders_from_memory, SODNet
from dcgan_funcs import fit_dcgan, generate, get_celeba_loader_from_memory, Generator, Discriminator, dcgan_weights_init 
from clf_funcs import fit, test, get_cifar10_loaders, get_mnist_loaders, FullyConnectedNet, SimpleConvNet

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121, mobilenet_v2, convnext_tiny


RUN_CLFS = True
clfs = ['FullyConnectedNet', 'SimpleConvNet', 'ResNet-50', 'DenseNet-121', 'MobileNet-v2', 'ConvNeXt-Tiny']
clfs = []
RUN_DCGAN = True
RUN_SODNET = True

def env_builder(name: str, num_classes: int, batch_size: int, test_batch_size: int): 
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
	elif name == 'ConvNeXt-Tiny':
		model = convnext_tiny()
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


if __name__ == '__main__':
	telemetry = {
		'model_name': [],
		'type':[],
		'epoch': [],
		'loss': [],
		'performance': [],
		'elapsed_time_real': [],
		'elapsed_time_perf': [],
		'elapsed_time_cuda': []
	}

	batch_size = 96
	test_batch_size = 128
	epochs = 8
	lr = 1e-2
	momentum = 0.9
	num_classes = 10
	log_interval = 200
	results_filepath = f'../../results/pytorch-{time.time_ns()}.csv'

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(f'CUDA enabled: {use_cuda}')
	start_cuda = torch.cuda.Event(enable_timing=True)
	end_cuda = torch.cuda.Event(enable_timing=True)

	for model_name in clfs:
		print(f'Benchmarks for {model_name} begin')

		model, train_dl, test_dl, loss_func = env_builder(model_name, num_classes, batch_size, test_batch_size)
		model = model.to(device)
		opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

		# training
		for epoch in range(1, epochs + 1):
			start_real = time.time_ns()
			start_perf = time.perf_counter_ns()
			start_cuda.record()
			train_history = fit(model, device, train_dl, loss_func, epoch, optimizer=opt, log_interval=log_interval, silent=False)
			end_real = time.time_ns()
			end_perf = time.perf_counter_ns()
			end_cuda.record()
			torch.cuda.synchronize()

			_, accuracy = test(model, device, test_dl, loss_func, silent=True)
			
			train_history = np.mean(train_history)

			telemetry['model_name'].append(model_name)
			telemetry['type'].append('training')
			telemetry['epoch'].append(epoch)
			telemetry['loss'].append(train_history)
			telemetry['performance'].append(accuracy)
			telemetry['elapsed_time_real'].append(end_real - start_real)
			telemetry['elapsed_time_perf'].append(end_perf - start_perf)
			telemetry['elapsed_time_cuda'].append(start_cuda.elapsed_time(end_cuda)  * 1e6)

		# inference
		start_real = time.time_ns()
		start_perf = time.perf_counter_ns()
		start_cuda.record()
		loss, accuracy = test(model, device, test_dl, loss_func, silent=True)
		end_real = time.time_ns()
		end_perf = time.perf_counter_ns()
		end_cuda.record()
		torch.cuda.synchronize()

		telemetry['model_name'].append(model_name)
		telemetry['type'].append('inference')
		telemetry['epoch'].append(1)
		telemetry['loss'].append(loss)
		telemetry['performance'].append(accuracy)
		telemetry['elapsed_time_real'].append(end_real - start_real)
		telemetry['elapsed_time_perf'].append(end_perf - start_perf)
		telemetry['elapsed_time_cuda'].append(start_cuda.elapsed_time(end_cuda)  * 1e6)
		pd.DataFrame(telemetry).to_csv(results_filepath, index=False)

		del model

	nc = 3
	nz = 100
	ngf = 64
	ndf = 64
	lr = 1e-4
	gen_batch_size = 1024

	netG = Generator(nc, nz, ngf).to(device)
	netD = Discriminator(nc, ndf).to(device)
	netG.apply(dcgan_weights_init)
	netD.apply(dcgan_weights_init)

	celeba_dl = get_celeba_loader_from_memory(batch_size=batch_size, root='../../datasets/celeba_tiny')

	loss_func = nn.BCELoss()
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

	print('Benchmarks for DCGAN begin')
	for epoch in range(1, epochs + 1):
		start_real = time.time_ns()
		start_perf = time.perf_counter_ns()
		start_cuda.record()
		gan_hist = fit_dcgan(netG, netD, device, celeba_dl, loss_func, epoch, optimizerG, optimizerD, nz, log_interval=log_interval)
		end_real = time.time_ns()
		end_perf = time.perf_counter_ns()
		end_cuda.record()
		torch.cuda.synchronize()

		for stat in gan_hist:
			gan_hist[stat] = np.mean(gan_hist[stat])	

		telemetry['model_name'].append('DCGAN')
		telemetry['type'].append('training')
		telemetry['epoch'].append(epoch)
		telemetry['loss'].append(f'{gan_hist["loss_G"]}|{gan_hist["loss_D"]}')
		telemetry['performance'].append(f'{gan_hist["D_x"]}|{gan_hist["D_G_z1"]}|{gan_hist["D_G_z2"]}')
		telemetry['elapsed_time_real'].append(end_real - start_real)
		telemetry['elapsed_time_perf'].append(end_perf - start_perf)
		telemetry['elapsed_time_cuda'].append(start_cuda.elapsed_time(end_cuda)  * 1e6)
		pd.DataFrame(telemetry).to_csv(results_filepath, index=False)

	# generation
	start_real = time.time_ns()
	start_perf = time.perf_counter_ns()
	start_cuda.record()
	_ = generate(netG, device, gen_batch_size, save=False)
	end_real = time.time_ns()
	end_perf = time.perf_counter_ns()
	end_cuda.record()
	torch.cuda.synchronize()

	telemetry['model_name'].append('DCGAN')
	telemetry['type'].append('generation')
	telemetry['epoch'].append(1)
	telemetry['loss'].append(-1)
	telemetry['performance'].append(-1)
	telemetry['elapsed_time_real'].append(end_real - start_real)
	telemetry['elapsed_time_perf'].append(end_perf - start_perf)
	telemetry['elapsed_time_cuda'].append(start_cuda.elapsed_time(end_cuda)  * 1e6)
	pd.DataFrame(telemetry).to_csv(results_filepath, index=False)


	print('Benchmarks for SODNet begin')
	train_dl, test_dl = get_adam_loaders_from_memory(8, cutoff=0.8, root='../../datasets/ADAM/Training1200')
	model = SODNet(3, 16).to(device)
	model = model.to(device)
	loss_func = nn.SmoothL1Loss(reduction="sum")
	optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

	for epoch in range(1, epochs + 1):

		start_real = time.time_ns()
		start_perf = time.perf_counter_ns()
		start_cuda.record()
		train_loss = fit_sodnet(model, device, train_dl, loss_func, optimizer)
		end_real = time.time_ns()
		end_perf = time.perf_counter_ns()
		end_cuda.record()
		torch.cuda.synchronize()

		telemetry['model_name'].append('SODNet')
		telemetry['type'].append('training')
		telemetry['epoch'].append(epoch)
		telemetry['loss'].append(train_loss)
		telemetry['performance'].append(-1)
		# IoU would have to be measured during training and therefore make the results incomparable
		telemetry['elapsed_time_real'].append(end_real - start_real)
		telemetry['elapsed_time_perf'].append(end_perf - start_perf)
		telemetry['elapsed_time_cuda'].append(start_cuda.elapsed_time(end_cuda)  * 1e6)
		pd.DataFrame(telemetry).to_csv(results_filepath, index=False)

		print('[%d]\tTrain loss: %.4f' % (epoch, train_loss))

	start_real = time.time_ns()
	start_perf = time.perf_counter_ns()
	start_cuda.record()
	eval_loss = test_sodnet(model, device, test_dl, loss_func)
	end_real = time.time_ns()
	end_perf = time.perf_counter_ns()
	end_cuda.record()
	torch.cuda.synchronize()

	telemetry['model_name'].append('SODNet')
	telemetry['type'].append('detection')
	telemetry['epoch'].append(1)
	telemetry['loss'].append(eval_loss)
	telemetry['performance'].append(-1)
	telemetry['elapsed_time_real'].append(end_real - start_real)
	telemetry['elapsed_time_perf'].append(end_perf - start_perf)
	telemetry['elapsed_time_cuda'].append(start_cuda.elapsed_time(end_cuda)  * 1e6)
	pd.DataFrame(telemetry).to_csv(results_filepath, index=False)

	print('[%d]\tEval loss: %.4f' % (1, eval_loss))