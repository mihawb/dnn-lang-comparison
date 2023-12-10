import tensorflow as tf
from clf_funcs import get_cifar10_data, get_mnist_loaders, classifier_overlay, combine_model, FullyConnectedNet, SimpleConvNetBuilder, PerfCounterCallback, GeneratorBuilder, DiscriminatorBulider, get_celeba_loader, train_step
import pandas as pd
import multiprocessing as mp
import numpy as np
import time


def setup():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)


def env_builder(name, config): 
	if name == 'FullyConnectedNet':
		model = FullyConnectedNet()
		train_ds, test_ds = get_mnist_loaders(config['batch_size'], config['test_batch_size'])
	elif name == 'SimpleConvNet':
		model = SimpleConvNetBuilder()
		train_ds, test_ds = get_mnist_loaders(config['batch_size'], config['test_batch_size'], flatten=False)
	elif name == 'ResNet-50':
		model = combine_model(config['inputs'], tf.keras.applications.ResNet50, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.resnet50.preprocess_input)
	elif name == 'DenseNet-121':
		model = combine_model(config['inputs'], tf.keras.applications.DenseNet121, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.densenet.preprocess_input)
	elif name == 'MobileNet-v2':
		model = combine_model(config['inputs'], tf.keras.applications.MobileNetV2, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.mobilenet_v2.preprocess_input)
	elif name == 'ConvNeXt-Small':
		model = combine_model(config['inputs'], tf.keras.applications.ConvNeXtSmall, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.convnext.preprocess_input)
	else:
		raise ValueError('Invalid model name')

	return model, train_ds, test_ds


def train_single_model(model_name, config, telemetry, child_conn):
	print(f'Benchmarks for {model_name} begin')
	setup()

	model, train_ds, test_ds = env_builder(model_name, config)
	optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr'], momentum=config['momentum'])
	model.compile(optimizer=optimizer, loss=config['loss_func'], metrics=['accuracy'])
	perfcounter = PerfCounterCallback(telemetry)


	# training 
	if isinstance(train_ds, tuple):
		train_history = model.fit(
			train_ds[0], train_ds[1],
			batch_size=config['batch_size'],
			validation_data=test_ds,
			validation_batch_size=config['test_batch_size'],
			epochs=config['epochs'],
			shuffle=True,
			callbacks=[perfcounter]
		)
	else:
		train_history = model.fit(
			train_ds,
			validation_data=test_ds,
			validation_batch_size=config['test_batch_size'],
			epochs=config['epochs'],
			callbacks=[perfcounter]
		)

	telemetry['model_name'].extend([model_name] * config['epochs'])
	telemetry['type'].extend(['training'] * config['epochs'])
	telemetry['loss'].extend(train_history.history['loss'])
	telemetry['performance'].extend(train_history.history['val_accuracy'])
	# epoch and elapsed_time handeled by PerfCounterCallback

	# inference
	if isinstance(test_ds, tuple):
		eval_history = model.evaluate(
			test_ds[0], test_ds[1],
			batch_size=config['test_batch_size'],
			callbacks=[perfcounter]
		)
	else:
		eval_history = model.evaluate(
			test_ds,
			batch_size=config['test_batch_size'],
			callbacks=[perfcounter]
		)

	telemetry['model_name'].append(model_name)
	telemetry['type'].append('inference')
	telemetry['loss'].append(eval_history[0])
	telemetry['performance'].append(eval_history[1])
	# epoch and elapsed_time handeled by PerfCounterCallback

	child_conn.send(telemetry)
	pd.DataFrame(telemetry).to_csv(f'../../results/tensorflow.csv', index=False)

	del model, train_ds, test_ds, train_history


def train_dcgan(config, telemetry, child_conn):
	print('Benchmarks for DCGAN begin')
	setup()

	celeba_ds = get_celeba_loader(config['batch_size'], root='../../datasets/celeba_trunc')
	modelG = GeneratorBuilder()
	modelD = DiscriminatorBulider()
	optG = tf.keras.optimizers.Adam(config['lr'])
	optD = tf.keras.optimizers.Adam(config['lr'])
	# both need to be of tf.Variable type for tf.Funciton train_step
	loss_func = config['loss_func_GAN']
	latent_vec_size = config['latent_vec_size']

	running_loss_G, running_loss_D = 0.0, 0.0
	running_D_x, running_D_G_z1, running_D_G_z2 = 0.0, 0.0, 0.0

	for epoch in range(1, config['epochs'] + 1):
		print('epoch', epoch)
		history = {
			'loss_G': [],
			'loss_D': [],
			'D_x': [],
			'D_G_z1': [],
			'D_G_z2': []
		}
		start = time.perf_counter_ns()
		for batch_idx, data in enumerate(celeba_ds):
			errG, errD, D_x, D_G_z1, D_G_z2 = train_step(modelG, modelD, optG, optD, data, loss_func)

			# batch telemetry
			running_loss_G += errG
			running_loss_D += errD
			running_D_x += D_x
			running_D_G_z1 += D_G_z1
			running_D_G_z2 += D_G_z2
			if batch_idx % config['log_interval'] == config['log_interval'] - 1:
				history['loss_G'].append(running_loss_G / config['log_interval'])
				history['loss_D'].append(running_loss_D / config['log_interval'])
				history['D_x'].append(running_D_x / config['log_interval'])
				history['D_G_z1'].append(running_D_G_z1 / config['log_interval'])
				history['D_G_z2'].append(running_D_G_z2 / config['log_interval'])

				running_loss_G, running_loss_D = 0.0, 0.0
				running_D_x, running_D_G_z1, running_D_G_z2 = 0.0, 0.0, 0.0 

				if not config['silent']:
					print('[%d][%d/%d]\tLoss_G: %.4f\tLoss_D: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, batch_idx, len(celeba_ds),
						 errG, errD, D_x, D_G_z1, D_G_z2))
					
		end = time.perf_counter_ns()
					
		# epoch telemetry
		for stat in history:
			history[stat] = np.sum(history[stat]) / len(history[stat])

		telemetry['model_name'].append('DCGAN')
		telemetry['type'].append('training')
		telemetry['epoch'].append(epoch)
		telemetry['loss'].append(f'{history["loss_G"]}|{history["loss_D"]}')
		telemetry['performance'].append(f'{history["D_x"]}|{history["D_G_z1"]}|{history["D_G_z2"]}')
		telemetry['elapsed_time'].append(end - start)

	# generation
	noise = tf.random.normal([config['test_batch_size'], config['latent_vec_size']])
	start = time.perf_counter_ns()
	_ = modelG(noise)
	end = time.perf_counter_ns()

	# generation telemetry
	telemetry['model_name'].append('DCGAN')
	telemetry['type'].append('generation')
	telemetry['epoch'].append(1)
	telemetry['loss'].append(-1)
	telemetry['performance'].append(-1)
	telemetry['elapsed_time'].append(end - start)

	child_conn.send(telemetry)
	pd.DataFrame(telemetry).to_csv(f'../../results/tensorflow.csv', index=False)

	del modelG, modelD, celeba_ds


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
		'latent_vec_size': 100
	}
		
	# for model_name in ['FullyConnectedNet', 'SimpleConvNet', 'ResNet-50', 'DenseNet-121', 'MobileNet-v2', 'ConvNeXt-Small']:
		# p = mp.Process(target=train_single_model, args=(model_name, config, telemetry, child_conn))
		# p.start()
		# telemetry = parent_conn.recv()
		# p.join()

	p = mp.Process(target=train_dcgan, args=(config, telemetry, child_conn))
	p.start()
	telemetry = parent_conn.recv()
	p.join()
		