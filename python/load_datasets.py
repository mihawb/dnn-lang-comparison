import numpy as np
import pickle


def load_mnist_imgs_and_labels(imgs_path, labels_path) -> tuple[np.ndarray, np.ndarray]:
	i_hand = open(imgs_path, 'rb')
	l_hand = open(labels_path, 'rb')

	i_hand.seek(4, 0) # skipping "magic" numbers
	l_hand.seek(4, 0)

	n_imgs = int.from_bytes(i_hand.read(4), 'big')

	imgs = np.frombuffer(i_hand.read(), np.uint8, offset=8)
	imgs = (255 - imgs) / 255
	imgs = imgs.reshape(n_imgs, 28 * 28)

	labels = np.frombuffer(l_hand.read(), np.uint8, offset=4)

	i_hand.close()
	l_hand.close()

	return imgs, labels


def load_cifar10_imgs_and_labels(pickle_path) -> tuple[np.ndarray, np.ndarray]:
	with open(pickle_path, 'rb') as phand:
		d = pickle.load(phand, encoding='bytes')

	r = d[b'data'][:, :1024] / 255
	g = d[b'data'][:, 1024:2048] / 255
	b = d[b'data'][:, 2048:] / 255
	
	cstream = np.stack((r,g,b))
	cstream = np.moveaxis(cstream, 0, -1)
	cstream = cstream.reshape(-1, 32, 32, 3)

	return cstream, np.array(d[b'labels'])