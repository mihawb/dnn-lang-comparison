import tensorflow as tf
from tf_funcs import get_cifar10_data, get_mnist_loaders, classifier_overlay, combine_model, FullyConnectedNet, SimpleConvNetBuilder, PerfCounterCallback
import pandas as pd
import multiprocessing as mp


def setup():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)


def env_builder(name, config): 
	if name == 'fcnet':
		model = FullyConnectedNet()
		train_ds, test_ds = get_mnist_loaders(config['batch_size'], config['test_batch_size'])
	elif name == 'scvnet':
		model = SimpleConvNetBuilder()
		train_ds, test_ds = get_mnist_loaders(config['batch_size'], config['test_batch_size'], flatten=False)
	elif name == 'resnet50':
		model = combine_model(config['inputs'], tf.keras.applications.ResNet50, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.resnet50.preprocess_input)
	elif name == 'densenet121':
		model = combine_model(config['inputs'], tf.keras.applications.DenseNet121, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.densenet.preprocess_input)
	elif name == 'mobilenet_v2':
		model = combine_model(config['inputs'], tf.keras.applications.MobileNetV2, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(tf.keras.applications.mobilenet_v2.preprocess_input)
	elif name == 'convnext_small':
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

	telemetry['mnames'].extend([model_name] * config['epochs'])
	telemetry['type'].extend(['training'] * config['epochs'])
	telemetry['loss'].extend(train_history.history['loss'])
	telemetry['acc'].extend(train_history.history['val_accuracy'])
	# eps and times handeled by PerfCounterCallback

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

	telemetry['mnames'].append(model_name)
	telemetry['type'].append('inference')
	telemetry['loss'].append(eval_history[0])
	telemetry['acc'].append(eval_history[1])
	# eps and times handeled by PerfCounterCallback

	child_conn.send(telemetry)
	pd.DataFrame(telemetry).to_csv(f'../results/tensorflow_results.csv', index=False)

	del model, train_ds, test_ds, train_history


if __name__ == '__main__':
	mp.set_start_method('spawn')
	parent_conn, child_conn = mp.Pipe()

	telemetry = {
		'mnames': [],
		'type': [],
		'eps': [],
		'loss': [],
		'acc': [],
		'times': []
	}

	config = {
		'batch_size': 32,
		'test_batch_size': 64,
		'epochs': 15,
		'lr': 1e-2,
		'momentum': 0.0,
		'inputs': tf.keras.layers.Input(shape=(32,32,3)),
		'loss_func': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	}
		
	for model_name in ('fcnet', 'scvnet', 'resnet50', 'densenet121', 'mobilenet_v2', 'convnext_small'):
		p = mp.Process(target=train_single_model, args=(model_name, config, telemetry, child_conn))
		p.start()
		telemetry = parent_conn.recv()
		p.join()
		