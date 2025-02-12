import sys
sys.path.append('..')
from load_datasets import load_celeba_images
from clf_funcs import setup, PerfCounterCallback

import time
import pathlib
import numpy as np
import pandas as pd

import tensorflow as tf


def get_celeba_loader(batch_size, image_size=64, root='../../datasets/celeba'):
    return tf.keras.utils.image_dataset_from_directory(
        pathlib.Path(root),
		shuffle=True,
        label_mode=None,
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size
	)


def get_celeba_loader_from_memory_old(batch_size, image_size=64, root='../../datasets/celeba'):
	# saddly this is an expensive abstraction due to tensorflow's implicit GPU data transfering
	dl = get_celeba_loader(batch_size, image_size=image_size, root=root)
	collected_batches = [batch for batch in dl]
	return collected_batches


def get_celeba_loader_from_memory(batch_size, image_size=64, root='../../datasets/celeba'):
	root += '/img_align_celeba'
	imgs = []

	for img in load_celeba_images(root):
		resized = img.resize((image_size, image_size))
		img_arr = np.array(resized).astype(np.float32) / 255 # uint8::max
		imgs.append(img_arr)

		# debug
		if len(imgs) % 5000 == 0: print(len(imgs))
		# if len(imgs) >= 25000: break

	with tf.device('CPU'):
		ds = tf.data.Dataset.from_tensor_slices(imgs).batch(batch_size)
	print('ds built')

	return ds


def GeneratorBuilder(latent_vec_size=100, feat_map_size=64):
	gen = tf.keras.Sequential()
	gen.add(tf.keras.layers.Dense(4*4*feat_map_size*8, use_bias=False, input_shape=(latent_vec_size,),
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	gen.add(tf.keras.layers.ReLU())
	gen.add(tf.keras.layers.Reshape((4,4,feat_map_size*8)))
	assert gen.output_shape == (None, 4,4,feat_map_size*8)

	# gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size * 8, (4,4), (1,1), padding="same", use_bias=False))
	# gen.add(tf.keras.layers.BatchNormalization())
	# gen.add(tf.keras.layers.ReLU())
	# assert gen.output_shape == (None, 4, 4, feat_map_size * 8)

	gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size * 4, (4,4), (2,2), padding="same", use_bias=False,
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	gen.add(tf.keras.layers.BatchNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02),
		beta_initializer=tf.keras.initializers.Zeros()))
	gen.add(tf.keras.layers.ReLU())
	assert gen.output_shape == (None, 8, 8, feat_map_size * 4)

	gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size * 2, (4,4), (2,2), padding="same", use_bias=False,
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	gen.add(tf.keras.layers.BatchNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02),
		beta_initializer=tf.keras.initializers.Zeros()))
	gen.add(tf.keras.layers.ReLU())
	assert gen.output_shape == (None, 16, 16, feat_map_size * 2)

	gen.add(tf.keras.layers.Conv2DTranspose(feat_map_size, (4,4), (2,2), padding="same", use_bias=False,
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	gen.add(tf.keras.layers.BatchNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02),
		beta_initializer=tf.keras.initializers.Zeros()))
	gen.add(tf.keras.layers.ReLU())
	assert gen.output_shape == (None, 32, 32, feat_map_size)

	gen.add(tf.keras.layers.Conv2DTranspose(3, (4,4), (2,2), padding="same", use_bias=False, activation='tanh',
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	assert gen.output_shape == (None, 64, 64, 3)

	return gen


def DiscriminatorBulider(feat_map_size=64):
	disc = tf.keras.Sequential()

	disc.add(tf.keras.layers.Conv2D(feat_map_size, (4,4), (2,2), padding="same", use_bias=False, input_shape=(64,64,3),
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	disc.add(tf.keras.layers.LeakyReLU(0.2))

	disc.add(tf.keras.layers.Conv2D(feat_map_size * 2, (4,4), (2,2), padding="same", use_bias=False,
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	disc.add(tf.keras.layers.BatchNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02),
		beta_initializer=tf.keras.initializers.Zeros()))
	disc.add(tf.keras.layers.LeakyReLU(0.2))

	disc.add(tf.keras.layers.Conv2D(feat_map_size * 4, (4,4), (2,2), padding="same", use_bias=False,
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	disc.add(tf.keras.layers.BatchNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02),
		beta_initializer=tf.keras.initializers.Zeros()))
	disc.add(tf.keras.layers.LeakyReLU(0.2))

	disc.add(tf.keras.layers.Conv2D(feat_map_size * 8, (4,4), (2,2), padding="same", use_bias=False,
		kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)))
	disc.add(tf.keras.layers.Flatten())
	disc.add(tf.keras.layers.Dense(1, activation="sigmoid", use_bias=False))

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


def train_dcgan(config, telemetry, child_conn=None):
	setup()

	print('Loading CELEBA')
	start = time.perf_counter_ns()
	celeba_ds = get_celeba_loader_from_memory(config['batch_size'], root='../../datasets/celeba_tiny')
	end = time.perf_counter_ns()

	telemetry['model_name'].append('CELEBA')
	telemetry['phase'].append('read')
	telemetry['epoch'].append(1)
	telemetry['loss'].append(-1)
	telemetry['performance'].append(-1)
	telemetry['elapsed_time'].append(end - start)

	print('CELEBA loaded')

	modelG = GeneratorBuilder()
	modelD = DiscriminatorBulider()
	optG = tf.keras.optimizers.Adam(config['lr'])
	optD = tf.keras.optimizers.Adam(config['lr'])
	# both need to be of tf.Variable phase for tf.Funciton train_step
	loss_func = config['loss_func_GAN']
	latent_vec_size = config['latent_vec_size']

	running_loss_G, running_loss_D = 0.0, 0.0
	running_D_x, running_D_G_z1, running_D_G_z2 = 0.0, 0.0, 0.0

	print('Benchmarks for DCGAN begin')
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
			# maybe use `tensor = tf.convert_to_tensor(data)` bc ds was created with CPU?
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
		telemetry['phase'].append('training')
		telemetry['epoch'].append(epoch)
		telemetry['loss'].append(f'{history["loss_G"]}|{history["loss_D"]}')
		telemetry['performance'].append(f'{history["D_x"]}|{history["D_G_z1"]}|{history["D_G_z2"]}')
		telemetry['elapsed_time'].append(end - start)

	# generation
	noise = tf.random.normal([config['gen_batch_size'], config['latent_vec_size']])
	start = time.perf_counter_ns()
	_ = modelG(noise, training=False)
	end = time.perf_counter_ns()

	# generation telemetry
	telemetry['model_name'].append('DCGAN')
	telemetry['phase'].append('generation')
	telemetry['epoch'].append(1)
	telemetry['loss'].append(-1)
	telemetry['performance'].append(-1)
	telemetry['elapsed_time'].append(end - start)

	# latency
	perfcounter = PerfCounterCallback(telemetry_ref=telemetry, latency_ref=[])
	for rep in range(config['epochs'] + config['latency_warmup_steps']):
		sample = tf.random.normal([1, config['latent_vec_size']])
		_ = modelG.predict(sample, callbacks=[perfcounter])

	telemetry['model_name'].extend(['DCGAN'] * config['epochs'])
	telemetry['phase'].extend(['latency'] * config['epochs'])
	telemetry['epoch'].extend([i+1 for i in range(config['epochs'])])
	telemetry['loss'].extend([-1] * config['epochs'])
	telemetry['performance'].extend([-1] * config['epochs'])
	telemetry['elapsed_time'].extend(perfcounter.latency_ref[config['latency_warmup_steps']:])

	pd.DataFrame(telemetry).to_csv(config['results_filename'], index=False)
	if child_conn is not None: child_conn.send(telemetry)

	del modelG, modelD, celeba_ds