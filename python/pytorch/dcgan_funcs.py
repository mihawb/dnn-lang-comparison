import sys
sys.path.append('..')
from load_datasets import load_celeba_images

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision


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


def get_celeba_loader_from_memory(batch_size, image_size=64, root='../../datasets/celeba'):
    # a near-zero-cost abstraction since pytorch does not implicity move data to GPU
	dl = get_celeba_loader(batch_size, image_size=image_size, root=root)
	# collected_batches = [(batch[0].clone().detach(), batch[1].clone().detach()) for batch in dl]
	collected_batches = [batch for batch in dl]
	return collected_batches


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