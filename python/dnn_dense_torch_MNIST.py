import torch
import torch.nn as nn
import numpy as np
from loadImgsAndLabels import loadloadImgsAndLabels

train_imgs, train_labels = loadloadImgsAndLabels(
	'../datasets/train-images.idx3-ubyte',
	'../datasets/train-labels.idx1-ubyte'
)

test_imgs, test_labels = loadloadImgsAndLabels(
	'../datasets/t10k-images.idx3-ubyte',
	'../datasets/t10k-labels.idx1-ubyte'
)