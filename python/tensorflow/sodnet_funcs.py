import sys
sys.path.append('..')
from load_datasets import load_adam_image
from clf_funcs import setup, PerfCounterCallback

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf


def build_adam_dataset(df, image_size=256):
	imgs, bboxes = [], []

	for idx in df.index:
		img, bbox = load_adam_image(df, idx)

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
		self.mpool = tf.keras.layers.MaxPool2D((2,2))


	def call(self, x):
		x = self.base1(x) + x
		x = self.base2(x)
		x = self.mpool(x)
		return x
	

def SODNetBuilder(in_channels, first_output_channels):
	return tf.keras.Sequential([
		tf.keras.Input(shape=(256, 256, 3)),

		ResBlock(in_channels, first_output_channels),
		ResBlock(first_output_channels, 2 * first_output_channels),
		ResBlock(2 * first_output_channels, 4 * first_output_channels),
		ResBlock(4 * first_output_channels, 8 * first_output_channels),
		
		tf.keras.layers.Conv2D(16 * first_output_channels, 3),
		tf.keras.layers.MaxPool2D((2,2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(2),
	])


def train_sodnet(config, telemetry, child_conn):
	print(f'Benchmarks for SODNet begin')
	setup()

	train_ds, test_ds = get_adam_loaders(config['batch_size_SOD'], config['test_batch_size_SOD'], cutoff=0.8)
	model = SODNetBuilder(3, 16)
	optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
	model.compile(optimizer=optimizer, loss=config['loss_func_SOD'])
	perfcounter = PerfCounterCallback(telemetry)

	train_history = model.fit(
		train_ds,
		epochs=config['epochs'],
		shuffle=True,
		callbacks=[perfcounter]
	)
	
	telemetry['model_name'].extend(['SODNet'] * config['epochs'])
	telemetry['phase'].extend(['training'] * config['epochs'])
	telemetry['loss'].extend(train_history.history['loss'])
	telemetry['performance'].extend([-1] * config['epochs']) # no validation to so as to be comparable to pytorch
	# epoch and elapsed_time handeled by PerfCounterCallback

	eval_history = model.evaluate(
		test_ds,
		batch_size=config['test_batch_size_SOD'],
		callbacks=[perfcounter]
	)

	telemetry['model_name'].append('SODNet')
	telemetry['phase'].append('detection')
	telemetry['loss'].append(eval_history)
	telemetry['performance'].append(-1)
	# epoch and elapsed_time handeled by PerfCounterCallback

	# latency
	batch = next(iter(test_ds))
	for rep in range(config['epochs'] + config['latency_warmup_steps']):
		sample = np.expand_dims(batch[0][rep % config['batch_size_SOD']], axis=0)
		start = time.perf_counter_ns()
		_ = model(sample, training=False)
		end = time.perf_counter_ns()

		if rep >= config['latency_warmup_steps']:
			telemetry['model_name'].append('SODNet')
			telemetry['phase'].append('latency')
			telemetry['epoch'].append(rep + 1 - config['latency_warmup_steps'])
			telemetry['loss'].append(-1)
			telemetry['performance'].append(-1)
			telemetry['elapsed_time'].append(end - start)

	child_conn.send(telemetry)
	pd.DataFrame(telemetry).to_csv(config['results_filename'], index=False)

	del model, train_ds, test_ds, train_history, eval_history