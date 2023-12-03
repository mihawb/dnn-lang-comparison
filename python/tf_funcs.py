import tensorflow as tf
from load_datasets import load_mnist_imgs_and_labels
from time import perf_counter_ns
import pathlib


def get_cifar10_data(preprocess=None):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

	if preprocess is not None:
		x_train, x_test = preprocess(x_train), preprocess(x_test)

	return (x_train, y_train), (x_test, y_test)


def get_mnist_loaders(batch_size, test_batch_size=None, flatten=True):
	if not test_batch_size: test_batch_size = batch_size * 2

	x_train, y_train = load_mnist_imgs_and_labels(
		'../datasets/mnist-digits/train-images-idx3-ubyte',
		'../datasets/mnist-digits/train-labels-idx1-ubyte'
	)

	x_test, y_test = load_mnist_imgs_and_labels(
		'../datasets/mnist-digits/t10k-images-idx3-ubyte',
		'../datasets/mnist-digits/t10k-labels-idx1-ubyte'
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


def get_celeba_loader(batch_size, image_size=64):
    return tf.keras.utils.image_dataset_from_directory(
        pathlib.Path('../datasets/celeba'),
		shuffle=True,
        label_mode=None,
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size
	)


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


def GeneratorBuilder(latent_vec_size=100, feat_map_size=64):
	gen = tf.keras.Sequential()
	gen.add(tf.keras.layers.Dense(4*4*feat_map_size*8, use_bias=False, input_shape=(latent_vec_size,)))
	gen.add(tf.keras.layers.ReLU())
	gen.add(tf.keras.layers.Reshape((4,4,feat_map_size*8)))
	assert gen.output_shape == (None, 4,4,feat_map_size*8)

	# gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size * 8, (4,4), (1,1), padding="same", use_bias=False))
	# gen.add(tf.keras.layers.BatchNormalization())
	# gen.add(tf.keras.layers.ReLU())
	# assert gen.output_shape == (None, 4, 4, feat_map_size * 8)

	gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size * 4, (4,4), (2,2), padding="same", use_bias=False))
	gen.add(tf.keras.layers.BatchNormalization())
	gen.add(tf.keras.layers.ReLU())
	assert gen.output_shape == (None, 8, 8, feat_map_size * 4)

	gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size * 2, (4,4), (2,2), padding="same", use_bias=False))
	gen.add(tf.keras.layers.BatchNormalization())
	gen.add(tf.keras.layers.ReLU())
	assert gen.output_shape == (None, 16, 16, feat_map_size * 2)

	gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size, (4,4), (2,2), padding="same", use_bias=False))
	gen.add(tf.keras.layers.BatchNormalization())
	gen.add(tf.keras.layers.ReLU())
	assert gen.output_shape == (None, 32, 32, feat_map_size)

	gen.add(tf.keras.layers.Conv2DTranspose(3, (4,4), (2,2), padding="same", use_bias=False, activation='tanh'))
	assert gen.output_shape == (None, 64, 64, 3)

	return gen


def DiscriminatorBulider(feat_map_size=64):
	disc = tf.keras.Sequential()

	disc.add(tf.keras.layers.Conv2D(feat_map_size, (4,4), (2,2), padding="same", use_bias=False, input_shape=[64,64,3]))
	disc.add(tf.keras.layers.LeakyReLU(0.2))

	disc.add(tf.keras.layers.Conv2D(feat_map_size * 2, (4,4), (2,2), padding="same", use_bias=False))
	disc.add(tf.keras.layers.BatchNormalization())
	disc.add(tf.keras.layers.LeakyReLU(0.2))

	disc.add(tf.keras.layers.Conv2D(feat_map_size * 4, (4,4), (2,2), padding="same", use_bias=False))
	disc.add(tf.keras.layers.BatchNormalization())
	disc.add(tf.keras.layers.LeakyReLU(0.2))

	disc.add(tf.keras.layers.Conv2D(feat_map_size * 8, (4,4), (2,2), padding="same", use_bias=False))
	disc.add(tf.keras.layers.Flatten())
	disc.add(tf.keras.layers.Dense(1, activation="sigmoid"))

	return disc


# This annotation causes the function to be "compiled".
@tf.function
def train_step(modelG, modelD, optG, optD, imgs_batch, loss_func, latent_vec_size=100):
	noise = tf.random.normal([imgs_batch.shape[0], latent_vec_size])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = modelG(noise, training=True)

		real_output = modelD(imgs_batch, training=True)
		fake_output = modelD(generated_images, training=True)
		D_x = tf.reduce_mean(real_output)
		D_G_z1 = tf.reduce_mean(fake_output)

		# generator loss
		gen_loss = loss_func(tf.ones_like(fake_output), fake_output)

		# discrimanator loss
		real_loss = loss_func(tf.ones_like(real_output), real_output)
		fake_loss = loss_func(tf.zeros_like(fake_output), fake_output)
		disc_loss = real_loss + fake_loss

	gradients_of_generator = gen_tape.gradient(gen_loss, modelG.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, modelD.trainable_variables)

	optG.apply_gradients(zip(gradients_of_generator, modelG.trainable_variables))
	optD.apply_gradients(zip(gradients_of_discriminator, modelD.trainable_variables))

	fake_output_graded = modelD(generated_images)
	D_G_z2 = tf.reduce_mean(fake_output_graded)

	return gen_loss, disc_loss, D_x, D_G_z1, D_G_z2