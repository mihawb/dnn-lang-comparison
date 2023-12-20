from clf_funcs import train_single_model
from dcgan_funcs import train_dcgan

import time
import multiprocessing as mp

import tensorflow as tf


if __name__ == '__main__':
	mp.set_start_method('spawn')
	parent_conn, child_conn = mp.Pipe()

	telemetry = {
		'model_name': [],
		'type': [],
		'epoch': [],
		'loss': [],
		'performance': [],
		'elapsed_time': []
	}

	config = {
		'batch_size': 96,
		'test_batch_size': 128,
		'epochs': 8,
		'lr': 1e-4,
		'momentum': 0.0,
		'inputs': tf.keras.layers.Input(shape=(32,32,3)),
		'loss_func': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
		'loss_func_GAN': tf.keras.losses.BinaryCrossentropy(from_logits=False),
		'log_interval': 200,
		'silent': False,
		'latent_vec_size': 100,
		'results_filename': f'../../results/tensorflow-{time.time_ns()}.csv'
	}
		
	for model_name in ['FullyConnectedNet', 'SimpleConvNet', 'ResNet-50', 'DenseNet-121', 'MobileNet-v2', 'ConvNeXt-Tiny']:
		p = mp.Process(target=train_single_model, args=(model_name, config, telemetry, child_conn))
		p.start()
		telemetry = parent_conn.recv()
		p.join()

	p = mp.Process(target=train_dcgan, args=(config, telemetry, child_conn))
	p.start()
	telemetry = parent_conn.recv()
	p.join()
		