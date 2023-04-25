import numpy as np


def loadloadImgsAndLabels(imgs_path, labels_path) -> tuple[np.ndarray, np.ndarray]:
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