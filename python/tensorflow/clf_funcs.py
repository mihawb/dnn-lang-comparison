import sys
sys.path.append('..')
from load_datasets import load_mnist_imgs_and_labels

import time
import pathlib
import numpy as np
import pandas as pd
from time import perf_counter_ns

import tensorflow as tf


def setup():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
				tf.config.experimental.set_virtual_device_configuration(
          			gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
				# i have literally no idea how LIMITING the memory usage suddenly makes 
				# everything fit into vram???? especially that the usage (on nvidia-smi)
				# stays the same just shy of full capacity 
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
	elif name == 'ConvNeXt-Tiny':
		model = combine_model(config['inputs'], tf.keras.applications.ConvNeXtTiny, classifier_overlay)
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

	# single sample latency
	if isinstance(test_ds, tuple):
		for rep in range(config['epochs']):
			sample = np.expand_dims(test_ds[0][rep], axis=0)
			start = time.perf_counter_ns()
			_ = model(sample, training=False)
			end = time.perf_counter_ns()

			telemetry['model_name'].append(model_name)
			telemetry['type'].append('latency')
			telemetry['loss'].append(-1)
			telemetry['performance'].append(-1)
			telemetry['epoch'].append(rep)
			telemetry['elapsed_time'].append(end - start)
	else:
		batch = next(iter(test_ds))[0]
		for rep in range(config['epochs']):
			sample = np.expand_dims(batch[rep], axis=0)
			start = time.perf_counter_ns()
			_ = model(sample, training=False)
			end = time.perf_counter_ns()

			telemetry['model_name'].append(model_name)
			telemetry['type'].append('latency')
			telemetry['loss'].append(-1)
			telemetry['performance'].append(-1)
			telemetry['epoch'].append(rep)
			telemetry['elapsed_time'].append(end - start)

	child_conn.send(telemetry)
	pd.DataFrame(telemetry).to_csv(config['results_filename'], index=False)

	del model, train_ds, test_ds, train_history


def get_cifar10_data(preprocess=None):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

	if preprocess is not None:
		x_train, x_test = preprocess(x_train), preprocess(x_test)

	return (x_train, y_train), (x_test, y_test)


def get_mnist_loaders(batch_size, test_batch_size=None, flatten=True):
	if not test_batch_size: test_batch_size = batch_size * 2

	x_train, y_train = load_mnist_imgs_and_labels(
		'../../datasets/mnist-digits/train-images-idx3-ubyte',
		'../../datasets/mnist-digits/train-labels-idx1-ubyte'
	)

	x_test, y_test = load_mnist_imgs_and_labels(
		'../../datasets/mnist-digits/t10k-images-idx3-ubyte',
		'../../datasets/mnist-digits/t10k-labels-idx1-ubyte'
	)

	if not flatten:
		x_train, x_test = map(
			lambda x: x.reshape(-1, 28, 28, 1),
			(x_train, x_test)
		)

	train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size)

	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	test_ds = test_ds.batch(test_batch_size)

	return train_ds, test_ds


def classifier_overlay(inputs):
	x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
	x = tf.keras.layers.Flatten()(x)
	# x = tf.keras.layers.Dense(1024, activation="relu")(x)
	# x = tf.keras.layers.Dense(512, activation="relu")(x)
	x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
	return x


def combine_model(inputs, predef_model, classifier, image_size=32):
	predef_model_materialised = predef_model(
		input_shape=(image_size, image_size, 3),
		include_top=False,
		weights=None
	)

	resize = tf.keras.layers.Resizing(image_size, image_size)(inputs)

	feature_extractor = predef_model_materialised(resize)
	classification_output = classifier(feature_extractor)
	combined = tf.keras.Model(inputs=inputs, outputs=classification_output)

	return combined


class PerfCounterCallback(tf.keras.callbacks.Callback):
	def __init__(self, telemetry_ref):
		super().__init__()
		self.telemetry_ref = telemetry_ref
		self.times = []
		self.eps = []
		self.training = False

	# for training
	def on_train_begin(self, logs=None):
		self.training = True

	def on_epoch_begin(self, epoch, logs=None):
		self.ep_start = perf_counter_ns()

	def on_epoch_end(self, epoch, logs=None):
		self.times.append(perf_counter_ns() - self.ep_start)
		self.eps.append(epoch + 1)

	def on_train_end(self, logs=None):
		self.telemetry_ref['epoch'].extend(self.eps)
		self.telemetry_ref['elapsed_time'].extend(self.times)
		self.training = False

	# for evaluation
	def on_test_begin(self, logs=None):
		self.test_start = perf_counter_ns()

	def on_test_end(self, logs=None):
		if self.training: return 
		self.telemetry_ref['elapsed_time'].append(perf_counter_ns() - self.test_start)
		self.telemetry_ref['epoch'].append(1)


class FullyConnectedNet(tf.keras.Model):

	def __init__(self, hidden_layers=[800], num_classes=10):
		super().__init__()
		self.hidden_layers = tf.keras.Sequential([
			tf.keras.layers.Dense(n, activation=tf.nn.relu) for n in hidden_layers
		])
		self.output_layer = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
		# TensorFlow best practice is to use softmax activation on last layer
		# combined with *Crossentrpy(from_logits=False) loss function
		# instead of LogSoftmax (like in PyTorch) since NLL loss func is not implemented in TF

	def call(self, x):
		x = self.hidden_layers(x)
		return self.output_layer(x)


# sequential builder, will most probably become obsolete
# since eventually I want to use subclassing api like in PyTorch 
def FullyConnectedNetBuilder(hidden_layers=[800], num_classes=10):
	layers = [tf.keras.layers.Dense(n, activation=tf.nn.relu) for n in hidden_layers]
	layers.append(tf.keras.layers.Dense(num_classes, activation='log_softmax'))
	return tf.keras.Sequential(layers)


def SimpleConvNetBuilder(num_classes=10):
	layers = [
		tf.keras.Input(shape=(28, 28, 1)),
		tf.keras.layers.Conv2D(16, kernel_size=5, padding="same", activation='relu'),
		tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
		tf.keras.layers.Conv2D(32, kernel_size=5, padding="same", activation='relu'),
		tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(500, activation='relu'),
		tf.keras.layers.Dense(num_classes, activation='softmax'),
	]
	return tf.keras.Sequential(layers)