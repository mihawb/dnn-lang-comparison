import sys
sys.path.append('..')
from load_datasets import load_mnist_imgs_and_labels

import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision


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


def fit_dcgan(generator, discriminator, device, loader, loss_func, epoch, optimizerG, optimizerD, latent_vec_size, log_interval=100, silent=False):
	real_label = 1.
	fake_label = 0.
	history = {
		'loss_G': [],
		'loss_D': [],
		'D_x': [],
		'D_G_z1': [],
		'D_G_z2': []
	}
	running_loss_G, running_loss_D = 0.0, 0.0
	running_D_x, running_D_G_z1, running_D_G_z2 = 0.0, 0.0, 0.0 

	for batch_idx, data in enumerate(loader, 0):

		discriminator.zero_grad()
		real_cpu = data[0].to(device)
		b_size = real_cpu.size(0)
		label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

		output = discriminator(real_cpu).view(-1)
		errD_real = loss_func(output, label)
		errD_real.backward()
		D_x = output.mean().item()

		noise = torch.randn(b_size, latent_vec_size, 1, 1, device=device)
		fake = generator(noise)
		label.fill_(fake_label)
		output = discriminator(fake.detach()).view(-1)
		errD_fake = loss_func(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()

		generator.zero_grad()
		label.fill_(real_label)
		output = discriminator(fake).view(-1)
		errG = loss_func(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()

		# telemetry
		running_loss_G += errG.item()
		running_loss_D += errD.item()
		running_D_x += D_x
		running_D_G_z1 += D_G_z1
		running_D_G_z2 += D_G_z2
		if batch_idx % log_interval == log_interval - 1:
			history['loss_G'].append(running_loss_G / log_interval)
			history['loss_D'].append(running_loss_D / log_interval)
			history['D_x'].append(running_D_x / log_interval)
			history['D_G_z1'].append(running_D_G_z1 / log_interval)
			history['D_G_z2'].append(running_D_G_z2 / log_interval)

			running_loss_G, running_loss_D = 0.0, 0.0
			running_D_x, running_D_G_z1, running_D_G_z2 = 0.0, 0.0, 0.0 

			if not silent:
				print('[%d][%d/%d]\tLoss_G: %.4f\tLoss_D: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, batch_idx, len(loader),
						 errG.item(), errD.item(), D_x, D_G_z1, D_G_z2))
				
	return history


def fit_sodnet(model, device, loader, loss_func, epoch, optimizer, log_interval=100, silent=False):
	model.train()
      
	running_loss = 0
	running_iou = 0
  
	for image_batch, bbox_batch in loader:
		image_batch, bbox_batch = image_batch.to(device), bbox_batch.to(device)

		optimizer.zero_grad()

		output = model(image_batch)

		loss = loss_func(output, bbox_batch)
		with torch.no_grad():
			iou_metric = iou_batch(output, bbox_batch, device=device)

		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		running_iou += iou_metric.item()

	return running_loss/len(loader.dataset), running_iou/len(loader.dataset)


def generate(generator, device, test_batch_size=64, latent_vec_size=100, latent_vecs_batch=None, save=False):
	# batch size not smaller than 64
	if latent_vecs_batch is None:
		latent_vecs_batch = torch.randn(test_batch_size, latent_vec_size, 1, 1, device=device)

	with torch.no_grad():
		res_imgs = generator(latent_vecs_batch).detach().cpu()

	if save:
		plt.imsave(
			f'../../results/pytorch_dcgan_results_{time.time_ns()}.png',
			np.transpose(
				torchvision.utils.make_grid(res_imgs, padding=5, normalize=True).cpu().numpy(),
				(1,2,0))
		)


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


def test_sodnet(model, device, loader, loss_func):
	model.eval()

	running_loss = 0
	running_iou = 0

	with torch.no_grad():
		for image_batch, bbox_batch in loader:
			image_batch, bbox_batch = image_batch.to(device), bbox_batch.to(device)

			output = model(image_batch)

			loss = loss_func(output, bbox_batch)
			iou_metric = iou_batch(output, bbox_batch, device=device)

			running_loss += loss.item()
			running_iou += iou_metric.item()

	return running_loss/len(loader.dataset), running_iou/len(loader.dataset)


def inception_fit(model, device, loader, loss_func, epoch, optimizer, log_interval=100, silent=False):
	'''
	Fit Inception parameters to train data (with auxiliary classifiers).
	'''
	model.train()
	logs = []
	running_loss = 0.0

	for batch_idx, (xb, yb) in enumerate(loader):
		xb, yb = xb.to(device), yb.to(device)

		# removing aggregated gradients
		optimizer.zero_grad()

		# feed forward and loss
		# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
		pred, aux_pred = model(xb)
		loss1 = loss_func(pred, yb)
		loss2 = loss_func(aux_pred, yb)
		loss = loss1 + 0.4*loss2

		# backprop nad applying gradients 
		loss.backward()
		optimizer.step()

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


def get_celeba_loader(batch_size, image_size=64, root='../../datasets/celeba'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds = torchvision.datasets.ImageFolder(root=root, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    return dl


class ADAMDataset(torch.utils.data.Dataset):
	def __init__(self, root, transform, fovea_df=None):
		self.root = root
		self.transform = transform
		if fovea_df is None:
			self.fovea_df = pd.read_csv(f'{root}/fovea_location.csv', index_col='ID')
		else:
			self.fovea_df = fovea_df


	def __getitem__(self, index):
		image_name = self.fovea_df.loc[index, 'imgName']
		image_path = f"{self.root}/{('AMD' if image_name.startswith('A') else 'Non-AMD')}/{image_name}"
		image = Image.open(image_path)
		bbox = self.fovea_df.loc[index, ['Fovea_X','Fovea_Y']].values.astype(float)
		image, bbox = self.transform((image, bbox))
		return image, bbox


	def __len__(self):
		return len(self.fovea_df)
	

def get_adam_loaders(batch_size, test_batch_size=None, cutoff=1, root='../../datasets/ADAM/Training1200'):
	if test_batch_size is None: test_batch_size = batch_size * 2

	transform = ToTensor() # no need for compose as augumentation is handled by importing script

	fovea_df = pd.read_csv(f'{root}/fovea_location.csv').drop(['ID'], axis=1)
	train_df, test_df = train_test_split(fovea_df, test_size=1-cutoff, shuffle=True)

	train_ds = ADAMDataset(root, transform, fovea_df=train_df.reset_index())
	test_ds = ADAMDataset(root, transform, fovea_df=test_df.reset_index())

	train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
	test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=2)

	return train_dl, test_dl


def dcgan_weights_init(model):
    '''
    From the DCGAN paper, the authors specify that all model weights shall be
    randomly initializedfrom a Normal distribution with `mean=0`, `stdev=0.02`
    '''
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class ToTensor:
    '''Convert the image to a Pytorch tensor with
    the channel as first dimenstion and values 
    between 0 to 1. Also convert the label to tensor
    with values between 0 to 1'''
    def __init__(self, scale_label=True):
        self.scale_label = scale_label

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1]
        w, h = image.size
        c_x, c_y = label

        image = torchvision.transforms.functional.to_tensor(image)

        if self.scale_label:
            label = c_x/w, c_y/h
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
  

class ToPILImage:
    '''Convert a tensor image to PIL Image. 
    Also convert the label to a tuple with
    values with the image units'''
    def __init__(self, unscale_label=True):
        self.unscale_label = unscale_label

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1].tolist()

        image = torchvision.transforms.functional.to_pil_image(image)
        w, h = image.size

        if self.unscale_label:
            c_x, c_y = label
            label = c_x*w, c_y*h

        return image, label
	

def centroid_to_bbox(centroids, w=50/256, h=50/256, device=None):
	x0_y0 = centroids - torch.tensor([w/2, h/2]).to(device)
	x1_y1 = centroids + torch.tensor([w/2, h/2]).to(device)
	return torch.cat([x0_y0, x1_y1], dim=1)


def iou_batch(output_labels, target_labels, device=None):
	output_bbox = centroid_to_bbox(output_labels, device=device)
	target_bbox = centroid_to_bbox(target_labels, device=device)
	return torch.trace(torchvision.ops.box_iou(output_bbox, target_bbox))


class FullyConnectedNet(nn.Module):
	
	def __init__(self, layers=[784, 800, 10]):
		super(FullyConnectedNet, self).__init__()
		self.layers = nn.ModuleList([nn.Linear(a, b, dtype=torch.float64) for a, b in zip(layers[:-1], layers[1:])])

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = F.relu(layer(x))
		x = self.layers[-1](x)
		return F.log_softmax(x, dim=1)
		# PyTorch best practice is to use LogSoftmax activation on output layer combined with NLL as loss function
		# or to return logits from FF and apply CrossEntropy loss function


class SimpleConvNet(nn.Module):

	def __init__(self, num_classes=10):
		super().__init__()
		self.conv1 = nn.Sequential(         
			nn.Conv2d(1, 16, 5, 1, 2, dtype=torch.float64),
			nn.ReLU(),                                       
			nn.MaxPool2d(2)
		)
		self.conv2 = nn.Sequential(         
			nn.Conv2d(16, 32, 5, 1, 2, dtype=torch.float64),
			nn.ReLU(),
			nn.MaxPool2d(2),  
		)
		self.dense = nn.Linear(32 * 7 * 7, 500, dtype=torch.float64)
		self.classifier = nn.Linear(500, num_classes, dtype=torch.float64)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.flatten(x, 1)
		x = F.relu(self.dense(x))
		return F.log_softmax(self.classifier(x), dim=1)


class Generator(nn.Module):
    def __init__(self, n_channels, latent_vec_size, feat_map_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vec_size, feat_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_map_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_map_size * 8, feat_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_map_size * 4, feat_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_map_size * 2, feat_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size),
            nn.ReLU(True),
			
            nn.ConvTranspose2d(feat_map_size, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
	

class Discriminator(nn.Module):
    def __init__(self, n_channels, feat_map_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, feat_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map_size, feat_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map_size * 2, feat_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map_size * 4, feat_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
	

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True) 
        )
        self.base2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.base1(x) + x
        x = self.base2(x)
        return x
    

class SODNet(nn.Module):
    def __init__(self, in_channels, first_output_channels):
        super().__init__()
        self.main = nn.Sequential(
            ResBlock(in_channels, first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(first_output_channels, 2 * first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(2 * first_output_channels, 4 * first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(4 * first_output_channels, 8 * first_output_channels),
            nn.MaxPool2d(2),
            nn.Conv2d(8 * first_output_channels, 16 * first_output_channels, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 16 * first_output_channels, 2)
        )

    def forward(self, x):
        return self.main(x)