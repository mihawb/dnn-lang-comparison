import sys
sys.path.append('..')
from load_datasets import load_mnist_imgs_and_labels

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet50, densenet121, mobilenet_v2, convnext_tiny


def env_builder(name: str, num_classes: int, batch_size: int, test_batch_size: int): 
	if name == 'FullyConnectedNet':
		model = FullyConnectedNet()
	elif name == 'SimpleConvNet':
		model = SimpleConvNet()
	elif name == 'ExtendedConvNet':
		model = SimpleConvNet(inner_expand=5)
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
	elif name in ['SimpleConvNet', 'ExtendedConvNet']:
		train_dl, _, test_dl = get_mnist_loaders(batch_size, test_batch_size, flatten=False)
		loss_func = F.nll_loss
	else:
		train_dl, test_dl = get_cifar10_loaders(batch_size, test_batch_size)
		loss_func = F.cross_entropy

	return model, train_dl, test_dl, loss_func


def fit(model, device, loader, loss_func, epoch, optimizer, log_interval=100, silent=False):
	'''
	Fit model's parameters to train data.
	Returns list of training losses for every `log_interval`th batch.
	'''
	model.train()
	logs = []
	running_loss = 0.0

	for batch_idx, (xb, yb) in enumerate(loader, 0):
		xb, yb = xb.to(device), yb.to(device)

		# removing aggregated gradients
		optimizer.zero_grad()

		# feed forward
		pred = model(xb)
		loss = loss_func(pred, yb)

		# backprop and applying gradients 
		loss.backward()
		optimizer.step()

		# telemetry
		running_loss += loss.item()
		if batch_idx % log_interval == log_interval - 1:
			avg_loss = running_loss / log_interval # log_interval == batch count for aggregation
			running_loss = 0.0
			logs.append(avg_loss)

			if not silent:
				print('[{}]\t[{}/{} ({:.0f}%)]\tLoss {:.4f}'.format(
					epoch, batch_idx, len(loader),
					100. * batch_idx / len(loader), avg_loss
				))

	return logs


def test(model, device, loader, loss_func, silent=False) -> tuple[float, float]:
	'''
	Test model's performance on validation or test data.
	Returns test loss and accuracy.
	'''
	model.eval()
	test_loss = 0
	correct_pred = 0

	# no gradient calculations, as it is not necessary here
	with torch.no_grad():
		for xb, yb in loader:
			xb, yb = xb.to(device), yb.to(device)
			pred = model(xb)
			test_loss += loss_func(pred, yb, reduction='sum').item()
			pred = pred.argmax(dim=1, keepdim=True)
			correct_pred += pred.eq(yb.view_as(pred)).sum().item()
	
	test_loss /= len(loader.dataset)
	if not silent:
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct_pred, len(loader.dataset),
			100. * correct_pred / len(loader.dataset)
		))

	return test_loss, correct_pred / len(loader.dataset)


def get_mnist_loaders(batch_size, test_batch_size=None, cutoff=1, flatten=True):
	if test_batch_size is None: test_batch_size = batch_size * 2

	x_train, y_train = load_mnist_imgs_and_labels(
		'../../datasets/mnist-digits/train-images-idx3-ubyte',
		'../../datasets/mnist-digits/train-labels-idx1-ubyte'
	)

	x_train, x_val = np.split(x_train, [int(len(x_train) * cutoff)])
	y_train, y_val = np.split(y_train, [int(len(y_train) * cutoff)])

	x_test, y_test = load_mnist_imgs_and_labels(
		'../../datasets/mnist-digits/t10k-images-idx3-ubyte',
		'../../datasets/mnist-digits/t10k-labels-idx1-ubyte'
	)

	if not flatten:
		x_train, x_test = map(
			lambda x: x.reshape(-1, 1, 28, 28),
			(x_train, x_test)
		)

	x_train, y_train, x_val, y_val, x_test, y_test = map(
		torch.tensor,
		(x_train, y_train, x_val, y_val, x_test, y_test)
	)

	x_train, x_val, x_test = map(
		lambda x: x.to(torch.float32),
		(x_train, x_val, x_test)
	)

	train_ds = TensorDataset(x_train, y_train)
	val_ds = TensorDataset(x_val, y_val)
	test_ds = TensorDataset(x_test, y_test)

	train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_dl = DataLoader(val_ds, batch_size=test_batch_size)
	test_dl = DataLoader(test_ds, batch_size=test_batch_size)

	return train_dl, val_dl, test_dl


def get_cifar10_loaders(batch_size, test_batch_size=None, image_size=32):
	if test_batch_size is None: test_batch_size = batch_size * 2

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	train_ds = torchvision.datasets.CIFAR10(root='../../datasets/cifar-10-py/', train=True, download=True, transform=transform)
	test_ds = torchvision.datasets.CIFAR10(root='../../datasets/cifar-10-py/', train=False, download=True, transform=transform)

	train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
	test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=2)

	return train_dl, test_dl


class FullyConnectedNet(nn.Module):
	
	def __init__(self, layers=[784, 800, 10]):
		super(FullyConnectedNet, self).__init__()
		self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(layers[:-1], layers[1:])])

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = F.relu(layer(x))
		x = self.layers[-1](x)
		return F.log_softmax(x, dim=1)
		# PyTorch best practice is to use LogSoftmax activation on output layer combined with NLL as loss function
		# or to return logits from FF and apply CrossEntropy loss function


class SimpleConvNet(nn.Module):
	def __init__(self, num_classes=10, inner_expand=0):
		super().__init__()
		self.conv1 = nn.Sequential(         
			nn.Conv2d(1, 16, 5, 1, 2),
			nn.ReLU(),                                       
			nn.MaxPool2d(2)
		)
		self.conv2 = nn.Sequential(         
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# ECVNet expand
			*[nn.Sequential(         
				nn.Conv2d(32, 32, 3, 1, 5),
				nn.ReLU(),
				nn.MaxPool2d(2),  
			) for _ in range(inner_expand)]
		)

		self.dense = nn.Sequential(
			nn.Linear(32 * 7 * 7, 512),
			nn.ReLU(),
			# ECVNet expand
			*[nn.Sequential(
				nn.Linear(int(512 / 2 ** (i - 1)), int(512 / 2 ** i)), 
				nn.ReLU()
			) for i in range(1, inner_expand + 1)]
		)

		self.classifier = nn.Linear(int(512 / 2 ** inner_expand), num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)

		x = torch.flatten(x, 1)
		x = self.dense(x)

		return F.log_softmax(self.classifier(x), dim=1)