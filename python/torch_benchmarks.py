import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50, densenet121, mobilenet_v2, convnext_small
import torch.optim as optim
from torch_funcs import fit, test, get_cifar10_loaders, get_mnist_loaders, FullyConnectedNet, SimpleConvNet
import pandas as pd


batch_size = 32
test_batch_size = 64
epochs = 15
lr = 1e-2
momentum = 0.9
num_classes = 10
log_interval = 300
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda"
                      if use_cuda
											else "cpu"
										)
print(f'CUDA enabled: {use_cuda}')


def env_builder(name: str): 
	if name == 'fcnet':
		model = FullyConnectedNet()
	elif name == 'scvnet':
		model = SimpleConvNet()
	elif name == 'resnet50':
		model = resnet50()
		model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
	elif name == 'densenet121':
		model = densenet121()
		model.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
	elif name == 'mobilenet_v2':
		model = mobilenet_v2()
		model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
	elif name == 'convnext_small':
		model = convnext_small()
		model.classifier[2] = nn.Linear(in_features=768, out_features=num_classes, bias=True)
	else:
		raise ValueError('Invalid model name')

	if name == 'fcnet':
		train_dl, _, test_dl = get_mnist_loaders(batch_size, test_batch_size)
		loss_func = F.nll_loss
	elif name == 'scvnet':
		train_dl, _, test_dl = get_mnist_loaders(batch_size, test_batch_size, flatten=False)
		loss_func = F.nll_loss
	else:
		train_dl, test_dl = get_cifar10_loaders(batch_size, test_batch_size)
		loss_func = F.cross_entropy

	return model, train_dl, test_dl, loss_func


telemetry = {
	'mnames': [],
	'type':[],
	'eps': [],
	'loss': [],
	'acc': [],
	'times': []
}


if __name__ == '__main__':
	# for model_name in ('fcnet', 'scvnet', 'resnet50', 'densenet121', 'mobilenet_v2', 'convnext_small'):
	for model_name in ('scvnet',):
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

			telemetry['mnames'].append(model_name)
			telemetry['type'].append('training')
			telemetry['eps'].append(epoch)
			telemetry['loss'].append(train_history[-1])
			telemetry['acc'].append(accuracy)
			telemetry['times'].append(start.elapsed_time(end))

		# inference
		start.record()
		loss, accuracy = test(model, device, test_dl, loss_func, silent=True)
		end.record()
		torch.cuda.synchronize()

		telemetry['mnames'].append(model_name)
		telemetry['type'].append('inference')
		telemetry['eps'].append(1)
		telemetry['loss'].append(loss)
		telemetry['acc'].append(accuracy)
		telemetry['times'].append(start.elapsed_time(end))
		pd.DataFrame(telemetry).to_csv(f'../results/pytorch_results.csv', index=False)

		del model
