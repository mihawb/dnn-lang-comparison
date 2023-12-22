import sys
sys.path.append('..')
from load_datasets import load_image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf


def build_adam_dataset(df, image_size=256):
	imgs, bboxes = [], []

	for idx in df.index:
		img, bbox = load_image(df, idx)

		img_arr = np.array(img)
		imgs.append(img_arr)

		bbox_arr = np.array(bbox)
		bboxes.append(bbox_arr)

	img_tensor = np.stack(imgs).astype(np.float32) / 255 # uint8::max
	bbox_tensor = np.stack(bboxes).astype(np.float32) / image_size # image size

	return tf.data.Dataset.from_tensor_slices((img_tensor, bbox_tensor))


def get_adam_loaders(batch_size, test_batch_size=None, image_size=256, cutoff=1.0, root='../../datasets/ADAM/Training1200'):
	if not test_batch_size: test_batch_size = batch_size * 2

	fovea_df = pd.read_csv(f'{root}/fovea_location.csv', index_col='ID')
	train_df, test_df = train_test_split(fovea_df, test_size=1-cutoff, shuffle=True)

	train_ds = build_adam_dataset(train_df, image_size).shuffle(buffer_size=1024).batch(batch_size)
	test_ds = build_adam_dataset(test_df, image_size).batch(test_batch_size)

	return train_ds, test_ds


class ResBlock(tf.keras.Model):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.base1 = tf.keras.Sequential([
			tf.keras.layers.Conv2D(in_channels, 3, padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU()
		])


		self.base2 = tf.keras.Sequential([
			tf.keras.layers.Conv2D(out_channels, 3, padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU()
		])


	def call(self, x):
		x = self.base1(x) + x
		x = self.base2(x)
		return x
	

def SODNetBuilder(in_channels, first_output_channels):
	return tf.keras.Sequential([
		tf.keras.Input(shape=(256, 256, 3)),

		ResBlock(in_channels, first_output_channels),
		tf.keras.layers.MaxPool2D((2,2)),

		ResBlock(first_output_channels, 2 * first_output_channels),
		tf.keras.layers.MaxPool2D((2,2)),

		ResBlock(2 * first_output_channels, 4 * first_output_channels),
		tf.keras.layers.MaxPool2D((2,2)),

		ResBlock(4 * first_output_channels, 8 * first_output_channels),
		tf.keras.layers.MaxPool2D((2,2)),
		
		tf.keras.layers.Conv2D(16 * first_output_channels, 3),
		tf.keras.layers.MaxPool2D((2,2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(2),
	])