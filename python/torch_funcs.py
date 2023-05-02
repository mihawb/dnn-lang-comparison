from load_datasets import load_mnist_imgs_and_labels

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision


def fit(model, device, loader, loss_func, epoch, lr=0.01, optimizer=None, log_interval=100, silent=False):
	'''
	Fit model's parameters to train data.
	'''
	model.train()
	logs = []
	running_loss = 0.0

	for batch_idx, (xb, yb) in enumerate(loader):
		xb, yb = xb.to(device), yb.to(device)

		# removing aggregated gradients
		if optimizer: 
			optimizer.zero_grad()
		else:
			model.zero_grad()		

		# backporpagation
		pred = model(xb)
		loss = loss_func(pred, yb)
		loss.backward()

		# applying gradients 
		if optimizer: 
			optimizer.step()
		else:
			for param in model.parameters():		
				param.data -= param.grad * lr

		# telemetry
		running_loss += loss.item()
		if batch_idx % log_interval == log_interval - 1:
			avg_loss = running_loss / log_interval # log_interval == batch count for aggregation
			running_loss = 0.0
			logs.append(avg_loss)

			if not silent:
				print('Train Epoch: {} -> batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx, batch_idx * len(xb), len(loader.dataset),
					100. * batch_idx / len(loader), avg_loss
				))

	return logs


def test(model, device, loader, loss_func, silent=False):
	'''
	Test model's performance on validation or test data.
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
	return correct_pred / len(loader.dataset)


def get_mnist_loaders(batch_size, test_batch_size=None, cutoff=1):
	if not test_batch_size: test_batch_size = batch_size * 2

	x_train, y_train = load_mnist_imgs_and_labels(
		'../datasets/mnist-digits/train-images.idx3-ubyte',
		'../datasets/mnist-digits/train-labels.idx1-ubyte'
	)

	x_train, x_val = np.split(x_train, [int(len(x_train) * cutoff)])
	y_train, y_val = np.split(y_train, [int(len(y_train) * cutoff)])

	x_test, y_test = load_mnist_imgs_and_labels(
		'../datasets/mnist-digits/t10k-images.idx3-ubyte',
		'../datasets/mnist-digits/t10k-labels.idx1-ubyte'
	)

	x_train, y_train, x_val, y_val, x_test, y_test = map(
		torch.tensor,
		(x_train, y_train, x_val, y_val, x_test, y_test)
	)

	train_ds = TensorDataset(x_train, y_train)
	val_ds = TensorDataset(x_val, y_val)
	test_ds = TensorDataset(x_test, y_test)

	train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_dl = DataLoader(val_ds, batch_size=test_batch_size)
	test_dl = DataLoader(test_ds, batch_size=test_batch_size)

	return train_dl, val_dl, test_dl


def get_cifar10_loaders(batch_size, test_batch_size=None):
	if not test_batch_size: test_batch_size = batch_size * 2

	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	train_ds = torchvision.datasets.CIFAR10(root='../datasets/cifar-10-py/', train=True, download=True, transform=transform)
	test_ds = torchvision.datasets.CIFAR10(root='../datasets/cifar-10-py/', train=False, download=True, transform=transform)

	train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
	test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=2)

	return train_dl, test_dl


class FullyConnectedNet(nn.Module):
	def __init__(self, layers=[784, 800, 10]):
		super(FullyConnectedNet, self).__init__()
		self.layers = nn.ModuleList([nn.Linear(a, b, dtype=torch.float64) for a, b in zip(layers[:-1], layers[1:])])


	def forward(self, x):
		for layer in self.layers[:-1]:
			x = F.relu(layer(x))
		x = self.layers[-1](x)
		return F.log_softmax(x, dim=1)
		# since log_softmax is returned, loss function has to be nll
		# in order to achieve cross entropy in the end 