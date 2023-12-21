import numpy as np
import pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# MNIST

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

# CIFAR-10

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

# ADAM

def load_image(df, idx, root='../../datasets/ADAM/Training1200'):
	image_name = df.loc[idx, 'imgName']
	data_type = 'AMD' if image_name.startswith('A') else 'Non-AMD'
	image_path = f'{root}/{data_type}/{image_name}'
	image = Image.open(image_path)
	bbox = (df.loc[idx, 'Fovea_X'], df.loc[idx, 'Fovea_Y'])
	return image, bbox


def show_image_with_bbox(df, image, bbox, idx=None, ax=None):
	w, h = (50, 50)
	c_x, c_y = bbox
	image = image.copy()
	ImageDraw.Draw(image).rectangle(((c_x-w//2, c_y-h//2), (c_x+w//2, c_y+h//2)), outline='green', width=2)
	if ax is not None:
		ax.imshow(image)
		if idx is not None: ax.set_title(df.loc[idx, 'imgName'])
	else: 
		plt.imshow(image)
		if idx is not None: plt.title(df.loc[idx, 'imgName'])


def show_image_with_two_bboxes(image, target_bbox, eval_bbox, title, ax=None):
	w, h = (50, 50)
	t_c_x, t_c_y = target_bbox
	e_c_x, e_c_y = eval_bbox
	image = image.copy()
	ImageDraw.Draw(image).rectangle(((t_c_x-w//2, t_c_y-h//2), (t_c_x+w//2, t_c_y+h//2)), outline='green', width=2)
	ImageDraw.Draw(image).rectangle(((e_c_x-w//2, e_c_y-h//2), (e_c_x+w//2, e_c_y+h//2)), outline='red', width=2)
	if ax is not None:
		ax.imshow(image)
		ax.set_title(title)
	else: 
		plt.imshow(image)
		plt.title(title)