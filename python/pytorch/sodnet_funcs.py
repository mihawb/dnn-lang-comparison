import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision


def fit_sodnet(model, device, loader, loss_func, optimizer):
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