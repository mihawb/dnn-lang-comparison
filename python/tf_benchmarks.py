import tensorflow as tf
from tf_funcs import get_cifar10_data, get_mnist_loaders, classifier_overlay, combine_model, FullyConnectedNet, PerfCounterCallback
from datetime import datetime
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    

batch_size = 32
test_batch_size = 64
epochs = 2
lr = 1e-2
momentum = 0.0
results_filename = f'tensorflow_results_batchsize{batch_size}_{str(datetime.now()).replace(" ", "_").replace(".", ":")}.csv'

inputs = tf.keras.layers.Input(shape=(32,32,3))
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def env_builder(name: str): 
	if name == 'fcnet':
		model = FullyConnectedNet()
		train_ds, test_ds = get_mnist_loaders(batch_size, test_batch_size)
	elif name == 'resnet50':
		model = combine_model(inputs, tf.keras.applications.ResNet50, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(batch_size, test_batch_size,
                                       tf.keras.applications.resnet50.preprocess_input)
	elif name == 'densenet121':
		model = combine_model(inputs, tf.keras.applications.DenseNet121, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(batch_size, test_batch_size,
                                       tf.keras.applications.densenet.preprocess_input)
	elif name == 'mobilenet_v2':
		model = combine_model(inputs, tf.keras.applications.MobileNetV2, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(batch_size, test_batch_size,
                                       tf.keras.applications.mobilenet_v2.preprocess_input)
	elif name == 'convnext_base':
		model = combine_model(inputs, tf.keras.applications.ConvNeXtBase, classifier_overlay)
		train_ds, test_ds = get_cifar10_data(batch_size, test_batch_size,
                                       tf.keras.applications.convnext.preprocess_input)
	else:
		raise ValueError('Invalid model name')

	return model, train_ds, test_ds


telemetry = {
	'mnames': [],
	'eps': [],
	'trloss': [],
	'acc': [],
	'times': []
}
perfcounter = PerfCounterCallback(telemetry)


for model_name in ('fcnet', 'resnet50', 'densenet121', 'mobilenet_v2', 'convnext_base'):
	print(f'Benchmarks for {model_name} begin')

	model, train_ds, test_ds = env_builder(model_name)
	model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

	if isinstance(train_ds, tuple):
		train_history = model.fit(
			train_ds[0], train_ds[1],
			batch_size=batch_size,
			validation_data=test_ds,
			validation_batch_size=test_batch_size,
			epochs=epochs,
			shuffle=True,
			callbacks=[perfcounter]
		)
	else:
		train_history = model.fit(
			train_ds,
			validation_data=test_ds,
			validation_batch_size=test_batch_size,
			epochs=epochs,
			callbacks=[perfcounter]
		)

	telemetry['mnames'].extend([model_name] * epochs)
	telemetry['trloss'].extend(train_history['loss'])
	telemetry['acc'].extend(train_history['val_accuracy'])
	# eps and times handeled by PerfCounterCallback

	pd.DataFrame(telemetry).to_csv(f'../results/{results_filename}', index=False)

	del model, train_ds, test_ds